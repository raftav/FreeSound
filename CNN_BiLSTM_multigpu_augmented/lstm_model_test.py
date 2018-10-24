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

    def __init__(self, features, seq_length, config,is_training):

        if is_training:
            batch_size = tf.shape(features)[0]
            keep_prob = config.keep_prob
        else:
            batch_size = 1
            keep_prob = 1.0

        with tf.variable_scope('cnn_filter'):
            cnn_filter=tf.get_variable('cnn_filter',[1,config.filter_size,1,config.filter_out_channels],initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

        cnn_features=tf.expand_dims(features,axis=3)

        cnn_features=tf.nn.conv2d(input=cnn_features,
                                    filter=cnn_filter,
                                    strides=[1,1,1,1],
                                    padding='VALID')

        cnn_features = tf.nn.relu(cnn_features)
        cnn_features = tf.nn.max_pool(cnn_features,ksize=[1,1,config.audio_feat_dimension-config.filter_size+1,1],strides=[1,1,1,1],padding='VALID')

        cnn_features = tf.transpose(cnn_features,perm=[0,1,3,2])
        cnn_features=tf.squeeze(cnn_features,axis=3)

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
                inputs=cnn_features,
                initial_states_fw=init_states_fw,
                initial_states_bw=init_states_bw,
                dtype=tf.float32,
                sequence_length=seq_length)

        #rnn_output = tf.slice(rnn_outputs, [0, tf.shape(rnn_outputs)[1] - 1, 0], [-1, -1, -1])
        rnn_last_output_fw=output_state_fw[-1].h
        #rnn_last_output_bw=output_state_bw[-1].h
        #print(rnn_last_output_fw)
        #print(rnn_last_output_bw)
        rnn_last_output_bw=tf.slice(rnn_outputs,[0,0,config.n_hidden],[-1,1,-1])
        rnn_last_output_bw=tf.squeeze(rnn_last_output_bw,axis=1)

        rnn_output = tf.concat((rnn_last_output_fw,rnn_last_output_bw),axis=-1)

        #print(rnn_output)

        with tf.variable_scope('output'):
            output_weights = tf.get_variable('weights', [2*config.n_hidden, config.audio_labels_dim],
                                                dtype=tf.float32,
                                                initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            output_biases = tf.get_variable('biases', shape=[config.audio_labels_dim], dtype=tf.float32,
                                            initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))

        rnn_output = tf.reshape(rnn_output, [-1, 2*config.n_hidden])

        output = tf.matmul(rnn_output, output_weights) + output_biases

        logits = tf.reshape(output, [batch_size, -1, config.audio_labels_dim])
        self._logits = logits


    @property
    def logits(self):
        return self._logits
