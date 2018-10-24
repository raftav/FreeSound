from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import time
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import lstm_model 

#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])
restore_epoch = int(sys.argv[2])

class Configuration(object):
	
	audio_feat_dimension = 72

	audio_labels_dim=20
	
	n_hidden=100
	num_layers=5

	num_test_examples=237

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

###################################
# Auxiliary functions
###################################

def read_my_file_format(filename_queue,feat_dimension=123):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return context_parsed['length'],sequence_parsed['audio_feat'],tf.to_int32(sequence_parsed['audio_labels'])

def input_pipeline_validation(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
    
    sequence_length, audio_features, audio_labels = read_my_file_format(filename_queue,feat_dimension=config.audio_feat_dimension)

    audio_features_batch, audio_labels_batch , seq_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length],
                                                    batch_size=1,
                                                    num_threads=10,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for tick in plt.gca().xaxis.get_majorticklabels():
    	tick.set_horizontalalignment("right")


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    '''
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
	'''
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('ConfusionMatrix_exp{}_epoch{}'.format(ExpNum,restore_epoch))


def test():

	config=Configuration()

	filename_test=['/DATA_NEW/rtavarone/Ecomode_Sentences_2018/RecurrentData/TEST/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(config.num_test_examples)]
	for f in filename_test:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	with tf.Graph().as_default():

		with tf.device('/cpu:0'):
			with tf.name_scope('validation_batch'):
				audio_features_test, audio_labels_test, seq_length_test = input_pipeline_validation(filename_test,config)
		
		with tf.device('/cpu:0'):
			with tf.variable_scope('model'):
				print('Building validation model:')
				test_model = lstm_model.Model(audio_features_test,audio_labels_test,seq_length_test,config,is_training=None)
				print('done.')

		# variables initializer
		init_op = tf.local_variables_initializer()

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=None)

		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			sess.run(init_op)
			print('Restoring variables...')
			saver.restore(sess,checkpoints_dir+'model_epoch'+str(restore_epoch)+'.ckpt')
			print('Model loaded')

			# start queue coordinator and runners
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)

			try:
				count=0
				accuracy=0.0
				labels=[]
				predictions=[]
				while not coord.should_stop():

					example_accuracy , test_label , test_prediction = sess.run([test_model.accuracy,
																			test_model.labels,
																			test_model.prediction])

					print('label[{}] prediction[{}] accuracy[{}]'.format(test_label,test_prediction,example_accuracy))
					labels.append(test_label[0,0])
					predictions.append(test_prediction[0,0])

					accuracy += example_accuracy

			except tf.errors.OutOfRangeError:
				accuracy /= config.num_test_examples
				print('accuracy = ',accuracy)
				labels = np.asarray(labels)
				predictions = np.asarray(predictions)

				cnf_matrix = confusion_matrix(labels, predictions)

				classes = []
				for i in range(19):
					classes.append( 'IPEC_{:03d}'.format(i+1) )
				classes.append('IREC_or_INEC')

				plot=plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

			finally:
				coord.request_stop()

def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  tf.app.run()