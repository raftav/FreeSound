import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import time
import sys
from tensorflow.python.client import timeline

import lstm_model
from utils import input_pipeline

#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])
restore_epoch = int(sys.argv[2])

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
		if 'number of hidden layers' in params:
			num_layers=int(params.split(':',1)[1].strip())
			print('num_layers =',num_layers)
		if 'number of hidden units' in params:
			n_hidden=int(params.split(':',1)[1].strip())
			print('n_hidden =',n_hidden)        
	if '##' not in lines[-1].split('\t')[0]:
		last_epoch = int(lines[-1].split('\t')[0])
	else:
		last_epoch = 1
		
	print('last epoch saved=',last_epoch)

	audio_feat_dimension = 123
	audio_labels_dim = 41
	num_epochs = 5000
	keep_prob=0.8
	num_gpus=4

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'
labels_set = pd.read_csv('/home/rtavaron/KaggleFreesound/data/labels.csv', header=0)
submission_file='submission_exp{}_epoch{}.csv'.format(ExpNum,restore_epoch)
out_fp=open(submission_file,'w')
out_fp.write('fname,label\n')

def test():

	config=Configuration()

	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/cpu:0'):
			filenames = tf.placeholder(tf.string, shape=[None])
			batch_size = tf.placeholder(tf.int64, shape=())

			dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=32,buffer_size=10000)
			dataset = dataset.map(input_pipeline._test_parser, num_parallel_calls=64)
			dataset = dataset.repeat(1)

			# dataset = dataset.batch(batch_size)
			dataset = dataset.padded_batch(batch_size, padded_shapes=([],
																	  [None, 123])).filter(lambda x, y: tf.equal(tf.shape(x)[0], tf.cast(batch_size,tf.int32)))
			
			dataset = dataset.prefetch(5000)

			iterator = dataset.make_initializable_iterator()

			sequence_length, features  = iterator.get_next()

		with tf.device('/cpu:0'):
			with tf.variable_scope('model'):
				print('Building val model:')
				val_model = lstm_model.Model(features, sequence_length, config,is_training=False)
				print('done.\n')
			val_logits=val_model.logits
			posteriors = tf.nn.softmax(val_logits)
			
			#prediction = tf.argmax(val_logits, axis=2)
			#correct = tf.equal(prediction, tf.to_int64(labels))
			#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		# variables initializer
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		# save and restore all the variables.
		saver = tf.train.Saver()

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			sess.run(init_op)
			saver.restore(sess,checkpoints_dir+'model_epoch'+str(restore_epoch)+'.ckpt')
			print('Model restored')

			TestingStartTime = time.time()
			filenames_test=input_pipeline.get_test_files()
			num_examples_test=len(filenames_test)
			print('num test examples : ',num_examples_test)

			sess.run(iterator.initializer, feed_dict={filenames: filenames_test,
															  batch_size: 1})
			val_example_number=0

			while True:
				try:
					print('Eval sequence {}'.format(val_example_number))

					test_posteriors = sess.run(posteriors)
					print('posteriors:')
					print(test_posteriors)
					print('')
					
					posteriors_indices = np.argsort(test_posteriors,axis=2)
					print('posteriors_indices:')
					print(posteriors_indices)
					print('')
					
					posteriors_indices = np.flip(posteriors_indices,axis=2)
					print('reverse posteriors_indices:')
					print(posteriors_indices)
					print('')

					for i in range(5):
						index=posteriors_indices[0,0,i]
						print('index {} posterior = {}'.format(index,test_posteriors[0,0,index]))

					filename=filenames_test[val_example_number].replace('/home/rtavaron/KaggleFreesound/data/features_test/','').replace('.tfrecords','.wav')
					print('processing file ',filename)
					outstring=filename+','

					labels=[]
					for i in range(3):
						string_label=labels_set[labels_set.index==posteriors_indices[0,0,i]].label.values[0]
						outstring+=string_label+' '

					outstring=outstring[:-1]
					outstring+='\n'
					out_fp.write(outstring)
					out_fp.flush()


					val_example_number+=1

				except tf.errors.OutOfRangeError:
					removed_files=['0b0427e2.wav','6ea0099f.wav','b39975f5.wav']
					for file in removed_files:
						outstring=file+','
						for i in range(3):
							string_label=labels_set[labels_set.index==i].label.values[0]
							outstring+=string_label+' '
						outstring=outstring[:-1]
						outstring+='\n'
						out_fp.write(outstring)
						
					out_fp.close()
					break











def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  tf.app.run()