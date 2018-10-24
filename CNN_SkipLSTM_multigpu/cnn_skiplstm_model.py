import tensorflow as tf
from utils import skip_rnn_cells

class Model(object):

	def __init__(self,features,seq_length,config,is_training):

		# batch size cannot be inferred from features shape because
		# it must be defined statically
		if is_training:
			batch_size=tf.shape(features)[0]
		else:
			batch_size=1

		# slope of the sigmoid for slope annealing trick
		#slope = tf.to_float(global_step / config.updating_step) * tf.constant(config.slope_annealing_rate) + tf.constant(1.0)
		#self._slope = slope
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

		with tf.variable_scope('forward_cells'):
			forward_cells=[]
			init_states_fw=[]
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{}'.format(i+1)):
					cell=skip_rnn_cells.SkipLSTMCell(num_units=config.n_hidden,layer_norm=True, update_bias=0.0)
					forward_cells.append(cell)
					init_states_fw.append(cell.trainable_initial_state(batch_size))

		with tf.variable_scope('backward_cells'):
			backward_cells=[]
			init_states_bw=[]
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{}'.format(i+1)):
					cell=skip_rnn_cells.SkipLSTMCell(num_units=config.n_hidden,layer_norm=True, update_bias=0.0)
					backward_cells.append(cell)
					init_states_bw.append(cell.trainable_initial_state(batch_size))



		with tf.variable_scope('forward_rnn'):
			# rnn_outputs , last_state_fw , last_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
			# 															cells_fw=forward_cells,
			# 															cells_bw=backward_cells,
			# 															inputs=cnn_features,
			# 															initial_states_fw=init_states_fw,
			# 															initial_states_bw=init_states_bw,
			# 															dtype=tf.float32,
			# 															sequence_length=seq_length)
			rnn_outputs , last_state_fw , last_state_bw = tf.nn.bidirectional_dynamic_rnn(
																		cells_fw=forward_cells,
																		cells_bw=backward_cells,
																		inputs=cnn_features,
																		initial_states_fw=init_states_fw,
																		initial_states_bw=init_states_bw,
																		dtype=tf.float32,
																		sequence_length=seq_length)
			updated_states = rnn_outputs.state_gate
			print('updated_states:')
			print(updated_states)
			print('')

			rnn_last_output_fw=last_state_fw[-1].h
			rnn_last_output_bw=last_state_bw[-1].h

			print('rnn_outputs_fw:')
			print(rnn_outputs_fw)
			print('')

		with tf.variable_scope("Output"):
			
			output_fw_weights = tf.get_variable('forward_weights',[config.n_hidden,config.num_classes],dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_bw_weights = tf.get_variable('backward_weights',[config.n_hidden,config.num_classes],dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_biases = tf.get_variable('biases',shape=[config.num_classes],dtype=tf.float32,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

			rnn_outputs_fw = tf.reshape(rnn_outputs_fw,[-1,config.n_hidden])
			rnn_outputs_bw = tf.reshape(rnn_outputs_bw,[-1,config.n_hidden])		
			
			output = tf.matmul(rnn_outputs_fw,output_fw_weights) + tf.matmul(rnn_outputs_bw,output_bw_weights) + output_biases
		
			logits = tf.reshape(output,[batch_size,-1,config.num_classes])

			all_states=tf.reduce_sum(updated_states)
			self._logits=logits
			self._all_states=all_states


	@property
	def logits(self):
		return self._logits

	@property
	def all_states(self):
		return self._all_states
