import tensorflow as tf
from tf_model import Memory

"""Following this tutorial:
https://github.com/EvolvedSquid/tutorials/tree/master/dqn
https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a
http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
"""


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class QNetwork(tf.keras.models.Model):
    def __init__(self, n_actions, conv_layer_units, dense_layer_units):
        super(QNetwork, self).__init__()

        self.conv_layers = self.buildConvLayers(conv_layer_units)
        self.stream_splitter_layer = tf.keras.layers.Lambda(lambda w: tf.split(w, 2, 3))
        self.value_stream_layers = self.buildDenseLayers(dense_layer_units, 1)
        self.advantage_stream_layers = self.buildDenseLayers(dense_layer_units, n_actions)
        self.reduce_mean_layer = tf.keras.layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

    @staticmethod
    def buildConvLayers(units):
        ret = []
        for unit in units:
            ret.append(tf.keras.layers.Conv2D(unit, kernel_size=(3, 3),
                                              strides=2,
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2.),
                                              padding="same", activation='relu', use_bias=False))
            ret.append(tf.keras.layers.MaxPooling2D())
            ret.append(tf.keras.layers.BatchNormalization())
        return ret

    @staticmethod
    def buildDenseLayers(units, output_dims):
        ret = [tf.keras.layers.Flatten()]
        for unit in units:
            ret.append(tf.keras.layers.Dense(unit, activation="relu", kernel_initializer="RandomNormal"))
        ret.append(tf.keras.layers.Dense(output_dims, activation="linear", kernel_initializer="RandomNormal"))

        return ret

    @tf.function
    def call(self, inputs):
        propagate = self.input_layer(inputs)
        for layer in self.conv_layers:
            propagate = layer(propagate)
        val_stream, adv_stream = self.stream_splitter_layer(propagate)
        for layer in self.value_stream_layers:
            val_stream = layer(val_stream)
        for layer in self.advantage_stream_layers:
            adv_stream = layer(adv_stream)

        # aggregation layer:
        return tf.keras.layers.Add()([val_stream,
                                      tf.keras.layers.Subtract()([adv_stream, self.reduce_mean_layer(adv_stream)])])


class DDQNetwork(object):
    """Deep-Q reinforcement learning model:
    'Dueling Network Architectures for Deep Reinforcement Learning' - Wang2016, arXiv

    convolution_layers: shared part of network;
        list of number of filters for each convolutional layer
    dense_layers: initial part of each of the two streams (advantage and value)
        list of number of fully-connected nodes (dense layers)
    """

    def __init__(self, n_actions, convolution_layers, dense_layers, delta_epsilon=0.01):
        # super(DDQNetwork, self).__init__()
        self.main_network = QNetwork(n_actions, convolution_layers, dense_layers)
        self.target_network = QNetwork(n_actions, convolution_layers, dense_layers)
        self.epsilon = 1
        self.delta_epsilon = delta_epsilon
        self.optimizer = tf.optimizers.Adam(1e-3)

    # def call(self, inputs):
    #     return self.main_network(inputs)

    def main_predict(self, state):
        return self.main_network.predict(state)

    def target_predict(self, state):
        return self.target_network.predict(state)

    def action_query(self, state):
        return self.predict(state)[0]

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
        
    def loss_fn(self, Q_s_a, logits):
        return tf.losses.mean_squared_error(Q_s_a, logits)

    def save_model(self, folder_name):
        self.main_network = tf.keras.models.save_model(folder_name + '/dqn.h5')
        self.target_network = tf.keras.models.save_model(folder_name + '/target_dqn.h5')

    def load_model(self, folder_name):
        self.main_network = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_network = tf.keras.models.load_model(folder_name + '/target_dqn.h5')