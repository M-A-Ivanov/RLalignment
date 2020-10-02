import numpy as np
import tensorflow as tf
import json
import time
import os
from config import (BATCH_SIZE, CLIP_REWARD, DISCOUNT_FACTOR,
                    EVAL_LENGTH, FRAMES_BETWEEN_EVAL, INPUT_SHAPE,
                    LEARNING_RATE, LOAD_FROM, LOAD_REPLAY_BUFFER,
                    MAX_EPISODE_LENGTH, MAX_NOOP_STEPS, MEM_SIZE,
                    MIN_REPLAY_BUFFER_SIZE, PRIORITY_SCALE, SAVE_PATH,
                    TENSORBOARD_DIR, TOTAL_FRAMES, UPDATE_FREQ, USE_PER,
                    WRITE_TENSORBOARD, IMAGE_DIR)
from ReplayBuffer import ReplayBuffer
# from simple_cofiguration import Environment
from DDQRLNetwork import DDQNetwork
from myLEEM import LEEM_remote


class Agent(object):
    """Implements a standard DDDQN agent"""

    def __init__(self,
                 dqn,
                 replay_buffer,
                 n_actions,
                 batch_size=8,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=200,  # starts to fall after
                 replay_buffer_start_size=500,
                 max_frames=2000,
                 use_per=True):
        """
        Arguments:
            dqn: A DQN (returned by the DQN function) to predict moves
            target_dqn: A DQN (returned by the DQN function) to predict target-q values.  This can be initialized in the same way as the dqn argument
            replay_buffer: A ReplayBuffer object for holding all previous experiences
            n_actions: Number of possible actions for the given environment
            input_shape: Tuple/list describing the shape of the pre-processed environment
            batch_size: Number of samples to draw from the replay memory every updating session
            history_length: Number of historical frames available to the agent
            eps_initial: Initial epsilon value.
            eps_final: The "half-way" epsilon value.  The epsilon value decreases more slowly after this
            eps_final_frame: The final epsilon value
            eps_evaluation: The epsilon value used during evaluation
            eps_annealing_frames: Number of frames during which epsilon will be annealed to eps_final, then eps_final_frame
            replay_buffer_start_size: Size of replay buffer before beginning to learn (after this many frames, epsilon is decreased more slowly)
            max_frames: Number of total frames the agent will be trained for
            use_per: Use PER instead of classic experience replay
        """

        self.n_actions = n_actions
        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer
        self.use_per = use_per

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        # DQN
        self.DQN = dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        """Get the appropriate epsilon value from a given frame number
        Arguments:
            frame_number: Global frame number (used for epsilon)
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif self.replay_buffer_start_size <= frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2 * frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            frame_number: Global frame number (used for epsilon)
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the frame number
        eps = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(2, self.n_actions+1)

        # Otherwise, query the DQN for an action
        q_vals = self.DQN.main_predict(state)[0]
        return q_vals.argmax()

    def update_target_network(self):
        """Update the target Q network"""
        self.DQN.update_target_network()

    def add_experience(self, action, frame, reward, clip_reward=True):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(action, frame, reward, clip_reward)

    def learn(self, batch_size, gamma, frame_number, priority_scale=1.0):
        """Sample a batch and use it to improve the DQN
        Arguments:
            batch_size: How many samples to draw for an update
            gamma: Reward discount
            frame_number: Global frame number (used for calculating importances)
            priority_scale: How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
        Returns:
            The loss between the predicted and target Q as a float
        """

        if self.use_per:
            (states, actions, rewards, new_states) \
                , importance, indices = self.replay_buffer.get_minibatch(batch_size=self.batch_size,
                                                                         priority_scale=priority_scale)
            importance = importance ** (1 - self.calc_epsilon(frame_number))
        else:
            states, actions, rewards, new_states = self.replay_buffer.get_minibatch(
                batch_size=self.batch_size, priority_scale=priority_scale)

        # Main DQN estimates best action in new states
        arg_q_max = self.DQN.main_predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.DQN.target_predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma * double_q)

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN.main_network(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions,
                                                            dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        self.DQN.save_model(folder_name)
        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current},
                                **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.DQN.load_model(folder_name)
        self.optimizer = self.DQN.optimizer

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta


# Create environment
if __name__ == "__main__":
    LEEM = LEEM_remote
    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    convolution_layers = [4, 8]
    dense_layers = [8, 16]
    # Build main and target networks
    DQN = DDQNetwork(LEEM.n_actions, convolution_layers, dense_layers, delta_epsilon=0.01)

    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
    agent = Agent(DQN, replay_buffer, LEEM.n_actions, batch_size=BATCH_SIZE, use_per=USE_PER)

    # Training and evaluation
    if LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

        # Apply information loaded from meta
        frame_number = meta['frame_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']

    # Main loop
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                # Training

                epoch_frame = 0
                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    LEEM.reset()
                    episode_reward_sum = 0
                    for _ in range(MAX_EPISODE_LENGTH):
                        # Get action
                        action = agent.get_action(frame_number, LEEM.state)

                        # Take step
                        processed_frame, reward, terminal = LEEM.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        # Add experience to replay memory
                        agent.add_experience(action=action,
                                             frame=processed_frame[:, :],
                                             reward=reward, clip_reward=CLIP_REWARD)

                        # Update agent
                        if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                            loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                  frame_number=frame_number, priority_scale=PRIORITY_SCALE)
                            loss_list.append(loss)

                        # Update target network
                        if frame_number % UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                            agent.update_target_network()

                        # Break the loop when the game is over
                        if terminal:
                            terminal = False
                            break

                    rewards.append(episode_reward_sum)

                    # Output the progress every 10 games
                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        print(
                            f'Game number: {str(len(rewards)).zfill(6)}  '
                            f'Frame number: {str(frame_number).zfill(8)}'
                            f'  Average reward: {np.mean(rewards[-10:]):0.1f}  '
                            f'Time taken: {(time.time() - start_time):.1f}s')

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0

                for _ in range(EVAL_LENGTH):
                    if terminal:
                        LEEM.reset()
                        life_lost = True
                        episode_reward_sum = 0
                        terminal = False

                    # Step action
                    _, reward, terminal, life_lost = LEEM.step(action)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward

                    # On game-over
                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum
                # Print score and write to tensorboard
                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                # Save model
                if len(rewards) > 300 and SAVE_PATH is not None:
                    agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                               rewards=rewards, loss_list=loss_list)
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards,
                       loss_list=loss_list)
            print('Saved.')
