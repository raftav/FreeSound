"""
Extended version of the BasicLSTMCell and BasicGRUCell in TensorFlow that allows to easily add custom inits,
normalization, etc.
"""
import tensorflow as tf
from tensorflow.python.ops import nn_ops

from utils import rnn_ops

LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, layer_norm=False):
        """
        Initialize the basic LSTM cell
        :param num_units: int, the number of units in the LSTM cell
        :param forget_bias: float, the bias added to forget gates
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        """
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._layer_norm = layer_norm

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Parameters of gates are concatenated into one multiply for efficiency.
            with tf.variable_scope('feed_forward_weights'):
                concat_bottom = rnn_ops.linear([inputs], 4 * self._num_units, bias=False,weights_init=tf.glorot_uniform_initializer())
            with tf.variable_scope('recurrent_weights'):
                concat_prev = rnn_ops.linear([h], 4 * self._num_units, bias=False,weights_init=tf.orthogonal_initializer())

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i_bottom, j_bottom, f_bottom, o_bottom = tf.split(value=concat_bottom, num_or_size_splits=4, axis=1)    
            i_prev, j_prev, f_prev, o_prev = tf.split(value=concat_prev, num_or_size_splits=4, axis=1)

            i=i_bottom + i_prev
            j=j_bottom + j_prev
            f=f_bottom + f_prev
            o=o_bottom + o_prev

            if self._layer_norm:
                i = rnn_ops.layer_norm(i, name="i")
                j = rnn_ops.layer_norm(j, name="j")
                f = rnn_ops.layer_norm(f,initial_bias_value=self._forget_bias, name="f")
                o = rnn_ops.layer_norm(o, name="o")

            new_c = (c * tf.sigmoid(f) + tf.sigmoid(i) * self._activation(j))
            if self._layer_norm:
                new_c = rnn_ops.layer_norm(new_c,name='new_c')

            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the BasicLSTMCell
        :param batch_size: number of samples per batch
        :return: LSTMStateTuple
        """
        def _create_initial_state(batch_size, state_size, trainable=True, initializer=tf.random_normal_initializer()):
            s = tf.get_variable('initial_state', shape=[1, state_size], dtype=tf.float32, trainable=trainable,
                                    initializer=initializer)
            state = tf.tile(s, tf.stack([batch_size] + [1]))
            return state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            with tf.variable_scope('initial_c'):
                initial_c = _create_initial_state(batch_size, self._num_units)
            with tf.variable_scope('initial_h'):
                initial_h = _create_initial_state(batch_size, self._num_units)
        return tf.contrib.rnn.LSTMStateTuple(initial_c, initial_h)


class MultiLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Stack of Skip LSTM cells. The selection binary output is computed from the state of the cell on top of
    the stack.
    """
    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, layer_norm=False):
        """
        Initialize the stack of Skip LSTM cells
        :param num_units: list of int, the number of units in each LSTM cell
        :param forget_bias: float, the bias added to forget gates
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        :param update_bias: float, initial value for the bias added to the update state gate
        """
        if not isinstance(num_units, list):
            num_units = [num_units]
        self._num_units = num_units
        self._num_layers = len(self._num_units)
        self._forget_bias = forget_bias
        self._activation = activation
        self._layer_norm = layer_norm

    @property
    def state_size(self):
        return [LSTMStateTuple(num_units, num_units) for num_units in self._num_units[:]] 

    @property
    def output_size(self):
        return self._num_units[-1]

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            cell_input = inputs
            new_states = []

            # Compute update candidates for all layers
            for idx in range(self._num_layers):
                with tf.variable_scope('layer_%d' % (idx + 1)):
                    c_prev, h_prev = state[idx]

                    # Parameters of gates are concatenated into one multiply for efficiency.
                    with tf.variable_scope('feed_forward_weights'):
                        concat_bottom = rnn_ops.linear([cell_input], 4 * self._num_units[idx], bias=False,weights_init=tf.glorot_uniform_initializer())
                    with tf.variable_scope('recurrent_weights'):
                        concat_prev = rnn_ops.linear([h_prev], 4 * self._num_units[idx], bias=False,weights_init=tf.orthogonal_initializer())

                    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                    
                    i_bottom, j_bottom, f_bottom, o_bottom = tf.split(value=concat_bottom, num_or_size_splits=4, axis=1)    
                    i_prev, j_prev, f_prev, o_prev = tf.split(value=concat_prev, num_or_size_splits=4, axis=1)

                    i=i_bottom + i_prev
                    j=j_bottom + j_prev
                    f=f_bottom + f_prev
                    o=o_bottom + o_prev

                    if self._layer_norm:
                        i = rnn_ops.layer_norm(i, name="i")
                        j = rnn_ops.layer_norm(j, name="j")
                        f = rnn_ops.layer_norm(f,initial_bias_value=self._forget_bias, name="f")
                        o = rnn_ops.layer_norm(o, name="o")

                    new_c = (c_prev * tf.sigmoid(f) + tf.sigmoid(i) * self._activation(j))
                    new_h = self._activation(new_c) * tf.sigmoid(o)

                    new_states.append(LSTMStateTuple(new_c, new_h))
                    cell_input = new_h

            new_output = new_h

            return new_output, new_states

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the MultiSkipLSTMCell
        :param batch_size: number of samples per batch
        :return: list of SkipLSTMStateTuple
        """
        initial_states = []
        for idx in range(self._num_layers):
            with tf.variable_scope('layer_%d' % (idx + 1)):
                with tf.variable_scope('initial_c'):
                    initial_c = rnn_ops.create_initial_state(batch_size, self._num_units[idx])
                with tf.variable_scope('initial_h'):
                    initial_h = rnn_ops.create_initial_state(batch_size, self._num_units[idx])
                initial_states.append(LSTMStateTuple(initial_c, initial_h))
        return initial_states
