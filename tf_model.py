# import tensorflow as tf
import random
import numpy as np
import math
import matplotlib.pylab as plt
from simple_cofiguration import Environment
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

"""
Adapted example from:
https://adventuresinmachinelearning.com/reinforcement-learning-tensorflow/

How AI gym environments look like:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/
"""
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001
GAMMA = 0.99
BATCH_SIZE = 20


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size

        self.states = None
        self.Q_s_a = None

        self.logits = None
        self.optimizer = None
        self.var_init = None

        self.define_model()

    def define_model(self):
        self.states = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self.Q_s_a = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)

        fc1 = tf.layers.dense(self.states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc2, self.num_actions)
        loss = tf.losses.mean_squared_error(self.Q_s_a, self.logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self.logits, feed_dict={self.states: state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self.logits, feed_dict={self.states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self.optimizer, feed_dict={self.states: x_batch, self.Q_s_a: y_batch})


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class Process:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0
        max_x = -100
        if self._render:
            self._env.create_graph()
        while True:
            if self._render:
                self._env.update_graph()
                self._env.save_graph()

            action = self._choose_action(state)

            next_state, reward, done = self._env.step(action)

            if next_state[0] <= 0.1:
                reward += 100
            elif next_state[0] <= 0.25:
                reward += 20
            elif next_state[0] <= 0.5:
                reward += 10
            if next_state[0] > max_x:
                max_x = next_state[0]
            # is the game complete? If so, set the next state to
            # None for storage sake
            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._env.remove_graph()
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


if __name__ == "__main__":

    num_states = 4  # = number of currents + detector readings
    num_actions = 6  # = ways to change currents (reduce, increase) x number of lenses

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)
    env = Environment(120., 20., -20.)

    with tf.Session() as sess:
        sess.run(model.var_init)
        gr = Process(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
        num_episodes = 20
        cnt = 0
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt + 1, num_episodes))
            gr.run()
            cnt += 1
        plt.plot(gr.reward_store)
        plt.show()
        plt.close("all")
        plt.plot(gr.max_x_store)
        plt.show()
