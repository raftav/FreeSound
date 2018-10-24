import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sys
from tensorflow.python.client import timeline

import lstm_model
from utils import input_pipeline

ExpNum = int(sys.argv[1])

class Configuration(object):
	logfile= open('TrainingExperiment'+str(ExpNum)+'.txt','r')
	lines = logfile.readlines()
	paramlines = [line for line in lines if line.startswith('##')]
	for params in paramlines:
		if 'learning rate' in params and 'update' not in params and 'decay' not in params:
			learning_rate = float (params.split(':',1)[1].strip() )
			print('learning_rate=',learning_rate)
		if 'batch size' in params:
			batch_size=int(params.split(':',1)[1].strip())
			print('batch size =',batch_size)
		if 'optimizer' in params:
		optimizer_choice = params.split(':',1)[1].strip()
		print('optimizer =',optimizer_choice)


	if '##' not in lines[-1].split('\t')[0]:
		last_epoch = int(lines[-1].split('\t')[0])
	else:
		last_epoch = 1
		
	print('last epoch saved=',last_epoch)

	audio_feat_dimension = 123

	audio_labels_dim = 41

	num_epochs = 5000

	n_hidden = 100
	num_layers = 5

	keep_prob=0.8

checkpoints_dir = 'checkpoints/exp' + str(ExpNum) + '/'

tensorboard_dir = 'tensorboard/exp' + str(ExpNum) + '/'

trainingLogFile = open('TrainingExperiment' + str(ExpNum) + '.txt', 'a')

def _parser(example_proto):
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(example_proto,
																	   context_features={
																		   "length": tf.FixedLenFeature([],
																										dtype=tf.int64)},
																	   sequence_features={
																		   "audio_feat": tf.FixedLenSequenceFeature(
																			   [123], dtype=tf.float32),
																		   "audio_labels": tf.FixedLenSequenceFeature(
																			   [], dtype=tf.float32)}
																	   )
	return context_parsed['length'], sequence_parsed['audio_feat'], tf.to_int32(sequence_parsed['audio_labels'])



def restore_train():

	config = Configuration()
	# training graph
	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/cpu:0'):
			filenames = tf.placeholder(tf.string, shape=[None])
			batch_size = tf.placeholder(tf.int64, shape=())
			keep_prob = tf.placeholder_with_default(1.0,shape=())

			dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=32,buffer_size=10000)
			dataset = dataset.map(_parser, num_parallel_calls=64)
			dataset = dataset.repeat(1)

			# dataset = dataset.batch(batch_size)
			dataset = dataset.padded_batch(batch_size, padded_shapes=([],
																	  [None, 123],
																	  [None]))

			dataset = dataset.prefetch(2000)

			iterator = dataset.make_initializable_iterator()

			sequence_length, features, labels = iterator.get_next()

		# audio features reconstruction
		with tf.device('/gpu:0'):
			with tf.variable_scope('model'):
				print('Building model:')
				model = lstm_model.Model(features, labels, sequence_length, config,keep_prob)
				print('done.\n')

		# variables initializer
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=10)

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			sess.run(init_op)
			saver.restore(sess,tf.train.latest_checkpoint(checkpoints_dir))
			print('variables loaded from ',tf.train.latest_checkpoint(checkpoints_dir))

			print('')
			print('## EXPERIMENT NUMBER ', ExpNum)
			print('## optimizer : ', config.optimizer_choice)
			print('## number of hidden layers : ', config.num_layers)
			print('## number of hidden units : ', config.n_hidden)
			print('## learning rate : ', config.learning_rate)
			print('## batch size : ', config.batch_size)

			step=sess.run(model.global_step)

			for epoch in range(config.last_epoch+1,config.num_epochs):
				filenames_train = input_pipeline.get_training_files(epoch)
				num_examples = len(filenames_train)
				print('Starting epoch ',epoch)
				print('Num training examples = ',num_examples)
				print('Num steps = ',num_examples/config.batch_size)

				sess.run(iterator.initializer, feed_dict={filenames: filenames_train,
													  batch_size: config.batch_size})

				EpochStartTime = time.time()
				partial_time=time.time()

				epoch_cost = 0.0

				while True:
					try:
						_, C, train_pred, train_label = sess.run([model.optimize, model.cost,
																  model.prediction,
																  model.labels],feed_dict={keep_prob:config.keep_prob})


						epoch_cost += C

						if (step % 10 == 0 or step == 1):
							print("step[{:7d}] cost[{:2.5f}] time[{}]".format(step, C,time.time()-partial_time))
							partial_time=time.time()

						# if (step % 500 == 0):
						#     # save training parameters
						#     save_path = saver.save(sess, checkpoints_dir + 'model_step' + str(step) + '.ckpt')
						#     print('Model saved!')

						step += 1

					except tf.errors.OutOfRangeError:
						break

				# End-of-training-epoch calculation here
				epoch_cost /= (num_examples / config.batch_size)

				print('Completed epoch {:d} at step {:d}. Average cost[{:.6f}]'.format(epoch, step, epoch_cost))
				print('Epoch training time (seconds) = ', time.time() - EpochStartTime)

				# evaluation every "plot_every_epoch" epochs
				plot_every_epoch = 1

				ValidationStartTime = time.time()

				# perform evaluation
				if ((epoch % plot_every_epoch) == 0):
					filenames_dev=input_pipeline.get_dev_files()
					num_examples_val=len(filenames_dev)

					sess.run(iterator.initializer, feed_dict={filenames: filenames_dev,
															  batch_size: 1})
					accuracy = 0
					val_example_number=0

					while True:
						try:
							example_accuracy, val_label, val_prediction = sess.run([model.accuracy,
																					model.labels,
																					model.prediction])
							accuracy += example_accuracy
							if val_example_number==0:
								print('label[{}] prediction[{}] accuracy[{}]'.format(val_label,
																					 val_prediction,
																					 example_accuracy))
								'''
								print('attention shape :',attention.shape)
								for value in range(attention.shape[1]):
									print('{:.5f}'.format(attention[0,value,0]))
								print('')
								print('attention sum = ',np.sum(attention,axis=1))
								'''
							val_example_number+=1

						except tf.errors.OutOfRangeError:
							break
					accuracy /= num_examples_val
					print('Validation accuracy : {} '.format(accuracy))

					save_path = saver.save(sess, checkpoints_dir + 'model_epoch' + str(epoch) + '.ckpt')
					print("model saved in file: %s" % save_path)

					outstring='{}\t{}\t{}\n'.format(epoch,epoch_cost,accuracy)
					trainingLogFile.write(outstring)
					trainingLogFile.flush()










def main(argv=None):  # pylint: disable=unused-argument
  restore_train()

if __name__ == '__main__':
  tf.app.run()