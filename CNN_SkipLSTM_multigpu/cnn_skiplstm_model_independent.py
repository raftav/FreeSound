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

		# stack of custom rnn cells
		num_units = [config.n_hidden for _ in range(config.num_layers)]

		with tf.variable_scope('forward_cells'):
			multi_cell_fw = skip_rnn_cells.MultiSkipGRUCell(num_units,layer_norm=True)
			initial_state_fw = multi_cell_fw.trainable_initial_state(batch_size)

		with tf.variable_scope('backward_cells'):
			multi_cell_bw = skip_rnn_cells.MultiSkipGRUCell(num_units,layer_norm=True)
			initial_state_bw = multi_cell_fw.trainable_initial_state(batch_size)


		with tf.variable_scope('forward_rnn'):
			rnn_outputs , last_state_fw  = tf.nn.dynamic_rnn(multi_cell_fw,
															cnn_features,
															initial_state=initial_state_fw,
															sequence_length=seq_length)
			
			rnn_outputs_fw, updated_states_fw = rnn_outputs.h, rnn_outputs.state_gate

			rnn_outputs_fw = rnn_outputs_fw * updated_states_fw
			rnn_outputs_fw = tf.reduce_sum(rnn_outputs_fw,axis=1) / tf.reduce_sum(updated_states_fw)

		with tf.variable_scope('backward_rnn'):
			input_reverse = tf.reverse_sequence(input=cnn_features,
											seq_lengths=seq_length,
											seq_axis=1, batch_axis=0)

			rnn_outputs , last_state_fw  = tf.nn.dynamic_rnn(multi_cell_bw,
															input_reverse,
															initial_state=initial_state_bw,
															sequence_length=seq_length)
			
			rnn_outputs_bw, updated_states_bw = rnn_outputs.h, rnn_outputs.state_gate

			rnn_outputs_bw = tf.reverse_sequence(input=rnn_outputs_bw,
											seq_lengths=seq_length,
											seq_axis=1, batch_axis=0)
			updated_states_bw = tf.reverse_sequence(input=updated_states_bw,
											seq_lengths=seq_length,
											seq_axis=1, batch_axis=0)

			rnn_outputs_bw = rnn_outputs_bw * updated_states_bw
			rnn_outputs_bw = tf.reduce_sum(rnn_outputs_bw,axis=1) / tf.reduce_sum(updated_states_bw)

		rnn_outputs = tf.concat((rnn_outputs_fw,rnn_outputs_bw),axis=1)


		with tf.variable_scope("Output"):
			
			output_weights = tf.get_variable('forward_weights',[2*config.n_hidden,config.num_classes],dtype=tf.float32,
			                        initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_biases = tf.get_variable('biases',shape=[config.num_classes],dtype=tf.float32,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

			#rnn_outputs = tf.reshape(rnn_outputs,[-1,2*config.n_hidden])	
			
			output = tf.matmul(rnn_outputs,output_weights) + output_biases
		
			logits = tf.reshape(output,[batch_size,-1,config.num_classes])

			all_states_fw=tf.reduce_sum(updated_states_fw)
			all_states_bw=tf.reduce_sum(updated_states_bw)
			all_states=all_states_bw+all_states_fw

			self._logits = logits
			self._all_states = all_states
			self._updates_fw = updated_states_fw
			self._updates_bw = updated_states_bw

	@property
	def logits(self):
		return self._logits

	@property
	def all_states(self):
		return self._all_states
	
	@property
	def updates_fw(self):
		return self._updates_fw
	
	@property
	def updates_bw(self):
		return self._updates_bw
	
		'''
		if is_training:

			# evaluate cost and optimize
			with tf.name_scope('cost'):

				all_states_fw=tf.reduce_sum(updated_states_fw)
				all_states_bw=tf.reduce_sum(updated_states_bw)
				all_states=all_states_bw+all_states_fw

				cross_entropy_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
				l2_loss = config.lambda_l2 * all_states
				loss = cross_entropy_loss + l2_loss

				#self._cross_entropy_loss = cross_entropy_loss
				#self._l2_loss = l2_loss
				self._cost = loss

				tf.summary.scalar('cost',self._cost)

			with tf.name_scope('optimizer'):
				learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,config.updating_step, config.learning_decay, staircase=True)
				self._learning_rate= learning_rate

				if 'momentum' in config.optimizer_choice:
					self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

				elif 'adam' in config.optimizer_choice:
					self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

				# gradient clipping
				gradients , variables = zip(*self._optimizer.compute_gradients(self._cost))
				clip_grad  = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients] 
				self._optimize = self._optimizer.apply_gradients(zip(clip_grad,variables),global_step=self._global_step)

		else:

			binary_mask = tf.sequence_mask(seq_length)
			prediction=tf.cast(tf.argmax(logits, axis=2),tf.int32)
			masked_prediction = tf.boolean_mask(prediction,binary_mask)
			masked_labels = tf.boolean_mask(labels,binary_mask)

			correct = tf.equal(masked_prediction,masked_labels)
			self._accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

			self._prediction = prediction
			self._labels = labels

			states_fw={}
			states_bw={}

			i=config.num_layers

			updated_states_fw = tf.boolean_mask(updated_states_fw,binary_mask)
			updated_states_bw = tf.boolean_mask(updated_states_bw,binary_mask)

			states_fw['z_{:d}'.format(i)] = updated_states_fw
			states_bw['z_{:d}'.format(i)] = updated_states_bw

			self._binary_states_fw = states_fw

			self._binary_states_bw = states_bw
		'''
		
