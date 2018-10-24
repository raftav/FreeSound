import tensorflow as tf
from utils import hgru

from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import layers

class Model(object):
	def _norm(self,inp, scope , norm_gain=1.0, norm_shift=0.0):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(norm_gain)
		beta_init = init_ops.constant_initializer(norm_shift)

		with tf.variable_scope(scope):
			tf.get_variable("gamma", shape=shape, initializer=gamma_init)
			tf.get_variable("beta", shape=shape, initializer=beta_init)
		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def __init__(self,features,seq_length,config,is_training):
		if is_training:
			batch_size=tf.shape(features)[0]
		else:
			batch_size=1

		# slope of the sigmoid for slope annealing trick
		slope = config.slope

		with tf.variable_scope('cnn_filter'):
			cnn_filter=tf.get_variable('cnn_filter',[5,5,1,1],initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

		cnn_features=tf.expand_dims(features,axis=3)
		cnn_features=tf.nn.conv2d(input=cnn_features,
									filter=cnn_filter,
									strides=[1,1,1,1],
									padding='SAME')

		cnn_features = tf.nn.relu(cnn_features)
		cnn_features = tf.nn.max_pool(cnn_features,ksize=[1,1,30,1],strides=[1,1,1,1],padding='SAME')

		cnn_features=tf.squeeze(cnn_features,axis=3)
		print('cnn_features')
		print(cnn_features)
		print('')

		# stack of custom rnn cells
		with tf.variable_scope('forward_cells'):
			cell_list_fw=[]
			zero_states_fw = []
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					rnn_cell= hgru.HGRUCell(config.n_hidden)
					rnn_cell.slope=slope
					h_init=tf.get_variable('h_init_state',[1,rnn_cell.state_size.h],
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
											dtype=tf.float32)
					h_init=tf.tile(h_init,[batch_size,1])

					z_init=tf.ones([1,rnn_cell.state_size.z])
					z_init=tf.tile(z_init,[batch_size,1])

				zero_states_fw.append(hgru.HardGatedStateTuple(h_init,z_init))
				cell_list_fw.append(rnn_cell)

			multi_cell_fw = hgru.MultiHGRUCell(cell_list_fw)
			initial_state_fw=tuple(zero_states_fw)
	
		with tf.variable_scope('backward_cells'):
			cell_list_bw=[]
			zero_states_bw = []

			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					rnn_cell= hgru.HGRUCell(config.n_hidden)
					rnn_cell.slope=slope

					h_init=tf.get_variable('h_init_state',[1,rnn_cell.state_size.h],
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
											dtype=tf.float32)
					h_init=tf.tile(h_init,[batch_size,1])

					z_init=tf.ones([1,rnn_cell.state_size.z])
					z_init=tf.tile(z_init,[batch_size,1])

				zero_states_bw.append(hgru.HardGatedStateTuple(h_init,z_init))
				cell_list_bw.append(rnn_cell)

			multi_cell_bw = hgru.MultiHGRUCell(cell_list_bw)
			initial_state_bw=tuple(zero_states_bw)

		# FORWARD RNN
		# use dynamic_rnn for training
		with tf.variable_scope('forward_rnn'):
			rnn_outputs_fw, last_state_fw = tf.nn.dynamic_rnn(multi_cell_fw,cnn_features,
														sequence_length=seq_length,
														initial_state=initial_state_fw)


		rnn_binary_states_fw = []
		for i in range(config.num_layers):
			rnn_binary_states_fw.append(rnn_outputs_fw[i].z)
		
		last_time_hidden_layers_fw = []
		for i in range(config.num_layers):
			last_time_hidden_layers_fw.append(last_state_fw[i].h)
		print('last_time_hidden_layers_fw')
		print(last_time_hidden_layers_fw)
		print('')

		with tf.variable_scope("Output_FW"):
			# output is another neural network with input from all hidden layers
			with tf.device('/gpu:0'):

				output_logits_fw = []
				for i in range(config.num_layers):
					#flatten batch and time dimensions of the layer i outputs
					rnn_outputs_layer_fw = tf.reshape(last_time_hidden_layers_fw[i],shape=[-1,last_time_hidden_layers_fw[i].get_shape().as_list()[-1]])

					# W_l^e
					output_embedding_matrix_fw = tf.get_variable("W{}".format(i),[ rnn_outputs_layer_fw.get_shape().as_list()[-1] , config.num_classes ],
												dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

					# output from a single layer
					output_logit_fw = tf.matmul(rnn_outputs_layer_fw, output_embedding_matrix_fw)
					output_logits_fw.append(output_logit_fw)


		# BACKWARD RNN
		input_reverse = tf.reverse_sequence(input=cnn_features,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)

		with tf.variable_scope('backward_rnn'):
			rnn_outputs_bw, last_state_bw = tf.nn.dynamic_rnn(multi_cell_bw,input_reverse,
															sequence_length=seq_length,
															initial_state=initial_state_bw)


		rnn_binary_states_bw = []
		for i in range(config.num_layers):
			# reverse all outputs before appending
			# rnn_output is a list (len=num_hidden_layer) of named tuple (h,z,z_tilda).
			temp_z = rnn_outputs_bw[i].z
			temp_z = tf.reverse_sequence(input=temp_z,
												seq_lengths=seq_length,
												seq_axis=1, batch_axis=0)
			rnn_binary_states_bw.append(temp_z)

		last_time_hidden_layers_bw=[]
		for i in range(config.num_layers):
			last_time_hidden_layers_bw.append(last_state_bw[i].h)

		print('last_time_hidden_layers_bw')
		print(last_time_hidden_layers_bw)
		print('')

		with tf.variable_scope("Output_BW"):
			# output is another neural network with input from all hidden layers
			with tf.device('/gpu:0'):

				output_logits_bw = []
				for i in range(config.num_layers):

					#flatten batch and time dimensions of the layer i outputs
					rnn_outputs_layer_bw = tf.reshape(last_time_hidden_layers_bw[i],shape=[-1,last_time_hidden_layers_bw[i].get_shape().as_list()[-1]])

					# W_l^e
					output_embedding_matrix_bw = tf.get_variable("W{}".format(i),[ rnn_outputs_layer_bw.get_shape().as_list()[-1] , config.num_classes ],
												dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

					# output from a single layer
					output_logit_bw = tf.matmul(rnn_outputs_layer_bw, output_embedding_matrix_bw)
					output_logits_bw.append(output_logit_bw)

		with tf.variable_scope("Output_FW_BW"):
			# Add independent forward and backward outputs
			output_logits=[]
			for i in range(config.num_layers):
				output_logits.append( output_logits_bw[i] + output_logits_fw[i] )

			# combined output from all layers
			output_logits = self._norm(tf.add_n(output_logits),"output_logits")
			output = tf.nn.relu(output_logits)
			
			# shape back to [batch_size, max_time, num_classes]
			logits = tf.reshape(output,shape=[batch_size,-1,config.num_classes])
			self._logits=logits

		if not is_training:

			states_fw={}
			for i in range(config.num_layers):
				states_fw['z_{:d}'.format(i)] = rnn_binary_states_fw[i]

			states_bw={}
			for i in range(config.num_layers):
				states_bw['z_{:d}'.format(i)] = rnn_binary_states_bw[i]

			self._binary_states_fw = states_fw
			self._binary_states_bw = states_bw


	@property
	def binary_states_fw(self):
		return self._binary_states_fw

	@property
	def binary_states_bw(self):
		return self._binary_states_bw

	@property
	def logits(self):
		return self._logits