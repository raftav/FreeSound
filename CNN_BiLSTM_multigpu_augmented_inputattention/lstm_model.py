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
			batch_size = 10
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


		with tf.variable_scope('input_masking_net'):
			with tf.variable_scope('embedding'):
				with tf.variable_scope('forward'):
					lstm_cell_forward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.masknet_num_neurons,
												forget_bias=1.0,
												activation=tf.tanh,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=keep_prob)
				with tf.variable_scope('backward'):
					lstm_cell_backward = tf.contrib.rnn.LayerNormBasicLSTMCell(config.masknet_num_neurons,
												forget_bias=1.0,
												activation=tf.tanh,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=keep_prob)

				with tf.variable_scope('attention_rnn'):
					input_attention, states_attention = tf.nn.bidirectional_dynamic_rnn(
														cell_fw=lstm_cell_forward,
														cell_bw=lstm_cell_backward,
														inputs=cnn_features,
														sequence_length=seq_length,
														initial_state_fw=lstm_cell_forward.zero_state(batch_size,dtype=tf.float32),
														initial_state_bw=lstm_cell_backward.zero_state(batch_size,dtype=tf.float32),
														dtype=tf.float32)

					input_attention = tf.concat(input_attention,2,name='recurrent_embedding')

			with tf.variable_scope('input_attention'):
				attention_weights = tf.get_variable('alpha_weights',[2*config.masknet_num_neurons,config.audio_feat_dimension],
													initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
				attention_bias = tf.get_variable('alpha_bias',[config.audio_feat_dimension],initializer=tf.zeros_initializer())

				input_attention = tf.reshape(input_attention,[-1,2*config.masknet_num_neurons])
				input_attention = tf.matmul(input_attention,attention_weights) + attention_bias
				input_attention = tf.reshape(input_attention,[batch_size,-1,config.audio_feat_dimension])

				alpha = tf.nn.softmax(input_attention,axis=2)

				masked_features = alpha * cnn_features
				self._masked_features = masked_features



		# lstm cells definition
		with tf.variable_scope('forward_cells'):
			forward_cells=[]
			init_states_fw=[]
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{}'.format(i+1)):
					cell=tf.contrib.rnn.LayerNormBasicLSTMCell(config.n_hidden,
												forget_bias=1.0,
												activation=tf.tanh,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=keep_prob)
					forward_cells.append( cell )
					init_states_fw.append( cell.zero_state(batch_size,dtype=tf.float32) )

		with tf.variable_scope('backward_cells'):
			backward_cells=[]
			init_states_bw=[]
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{}'.format(i+1)):
					cell=tf.contrib.rnn.LayerNormBasicLSTMCell(config.n_hidden,
												forget_bias=1.0,
												activation=tf.tanh,
												layer_norm=True,
												norm_gain=1.0,
												norm_shift=0.0,
												dropout_keep_prob=keep_prob)
					backward_cells.append( cell )
					init_states_bw.append( cell.zero_state(batch_size,dtype=tf.float32) )

		with tf.variable_scope('RNN'):
			rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
				cells_fw=forward_cells,
				cells_bw=backward_cells,
				inputs=masked_features,
				initial_states_fw=init_states_fw,
				initial_states_bw=init_states_bw,
				dtype=tf.float32,
				sequence_length=seq_length)

		rnn_last_output_fw=output_state_fw[-1].h
		rnn_last_output_bw=tf.slice(rnn_outputs,[0,0,config.n_hidden],[-1,1,-1])
		rnn_last_output_bw=tf.squeeze(rnn_last_output_bw,axis=1)
		rnn_output = tf.concat((rnn_last_output_fw,rnn_last_output_bw),axis=-1)


		with tf.variable_scope('output'):
			output_weights = tf.get_variable('weights', [2*config.n_hidden, config.audio_labels_dim],
												dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
			output_biases = tf.get_variable('biases', shape=[config.audio_labels_dim], dtype=tf.float32,
											initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))

		output = tf.reshape(rnn_output, [-1, 2*config.n_hidden])

		output = tf.matmul(output, output_weights) + output_biases

		logits = tf.reshape(output, [batch_size, -1, config.audio_labels_dim])
		self._logits = logits


	@property
	def logits(self):
		return self._logits
