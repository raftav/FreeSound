import tensorflow as tf
import sys
from utils import basic_rnn_cells

def variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var_name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class Model(object):

    def __init__(self, features, labels, seq_length, config,keep_prob):

        batch_size = tf.shape(features)[0]

        global_step = tf.Variable(0, trainable=False)
        self._global_step = global_step

        # lstm cells definition
        with tf.variable_scope('forward_cells'):
            forward_cells=[]
            init_states_fw=[]
            for i in range(config.num_layers):
                with tf.variable_scope('layer_{}'.format(i+1)):
                    cell=basic_rnn_cells.BasicLSTMCell(config.n_hidden,activation=tf.tanh)
                    forward_cells.append( tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob,variational_recurrent=True,dtype=tf.float32) )
                    init_states_fw.append( cell.trainable_initial_state(batch_size) )

        with tf.variable_scope('backward_cells'):
            backward_cells=[]
            init_states_bw=[]
            for i in range(config.num_layers):
                with tf.variable_scope('layer_{}'.format(i+1)):
                    cell=basic_rnn_cells.BasicLSTMCell(config.n_hidden,activation=tf.tanh)
                    backward_cells.append( tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob,variational_recurrent=True,dtype=tf.float32) )
                    init_states_bw.append( cell.trainable_initial_state(batch_size) )

        with tf.variable_scope('RNN'):
            rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=forward_cells,
                cells_bw=backward_cells,
                inputs=features,
                initial_states_fw=init_states_fw,
                initial_states_bw=init_states_bw,
                dtype=tf.float32,
                sequence_length=seq_length)

        #rnn_output = tf.slice(rnn_outputs, [0, tf.shape(rnn_outputs)[1] - 1, 0], [-1, -1, -1])
        rnn_last_output_fw=output_state_fw[-1].h
        rnn_last_output_bw=output_state_bw[-1].h
        print(rnn_last_output_fw)
        print(rnn_last_output_bw)

        rnn_output = tf.concat((rnn_last_output_fw,rnn_last_output_bw),axis=-1)

        print(rnn_output)

        with tf.variable_scope('output'):
            output_weights = tf.get_variable('weights', [2*config.n_hidden, config.audio_labels_dim],
                                                dtype=tf.float32,
                                                initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            output_biases = tf.get_variable('biases', shape=[config.audio_labels_dim], dtype=tf.float32,
                                            initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))

        rnn_output = tf.reshape(rnn_output, [-1, 2*config.n_hidden])

        output = tf.matmul(rnn_output, output_weights) + output_biases

        logits = tf.reshape(output, [batch_size, -1, config.audio_labels_dim])

        with tf.name_scope('cost'):
            self._cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        with tf.name_scope('optimizer'):
            # learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
            #                    config.updating_step, config.learning_decay, staircase=True)

            
            self._learning_rate = tf.Variable(config.learning_rate)

            if "momentum" in config.optimizer_choice:
                self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=0.9)
            elif "adam" in config.optimizer_choice:
                self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            else:
                print("Optimizer must be either momentum or adam. Closing.")
                sys.exit()

            # gradient clipping
            gradients, variables = zip(*self._optimizer.compute_gradients(self._cost))
            clip_grad = [None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in gradients]
            self._optimize = self._optimizer.apply_gradients(zip(clip_grad, variables),
                                                             global_step=self._global_step)

        posteriors = tf.nn.softmax(logits)
        prediction = tf.argmax(logits, axis=2)
        correct = tf.equal(prediction, tf.to_int64(labels))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self._posteriors = posteriors
        self._accuracy = accuracy
        self._labels = labels
        self._prediction = prediction


    @property
    def cost(self):
        return self._cost

    @property
    def optimize(self):
        return self._optimize

    @property
    def attention(self):
        return self._attention

    @property
    def posteriors(self):
        return self._posteriors

    @property
    def correct(self):
        return self._correct

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def labels(self):
        return self._labels

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def global_step(self):
        return self._global_step

    @property
    def prediction(self):
        return self._prediction
