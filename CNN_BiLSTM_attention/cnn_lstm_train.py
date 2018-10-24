# Avoid printing tensorflow log messages
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sys

import lstm_model
from utils import input_pipeline

#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])


class Configuration(object):
    learning_rate = float(sys.argv[2])
    batch_size = int(sys.argv[3])
    optimizer_choice = sys.argv[4]

    audio_feat_dimension = 123

    audio_labels_dim = 41

    num_epochs = 5000

    n_hidden = 100
    num_layers = 5

    keep_prob=0.8


checkpoints_dir = 'checkpoints/exp' + str(ExpNum) + '/'

tensorboard_dir = 'tensorboard/exp' + str(ExpNum) + '/'

trainingLogFile = open('TrainingExperiment' + str(ExpNum) + '.txt', 'w')

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


#################################
# Training module
#################################
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


def train():
    config = Configuration()

    # training graph
    with tf.Graph().as_default():

        # extract batch examples
        with tf.device('/cpu:0'):
            filenames = tf.placeholder(tf.string, shape=[None])
            batch_size = tf.placeholder(tf.int64, shape=())

            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(_parser, num_parallel_calls=10)
            dataset = dataset.repeat(1)

            # dataset = dataset.batch(batch_size)
            dataset = dataset.padded_batch(batch_size, padded_shapes=([],
                                                                      [None, 123],
                                                                      [None])).prefetch(buffer_size=10)

            iterator = dataset.make_initializable_iterator()

            sequence_length, features, labels = iterator.get_next()

        # audio features reconstruction
        with tf.device('/gpu:0'):
            with tf.variable_scope('model'):
                print('Building model:')
                train_model = lstm_model.Model(features, labels, sequence_length, config,is_training=True)
                print('done.\n')

        with tf.device('/cpu:0'):
            with tf.variable_scope('model',reuse=True):
                print('Building model:')
                val_model = lstm_model.Model(features, labels, sequence_length, config,is_training=False)
                print('done.\n')

        # variables initializer
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=10)

        # start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:

            sess.run(init_op)

            print('')
            print('## EXPERIMENT NUMBER ', ExpNum)
            trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))

            print('## optimizer : ', config.optimizer_choice)
            trainingLogFile.write('## optimizer : {:s} \n'.format(config.optimizer_choice))

            print('## number of hidden layers : ', config.num_layers)
            trainingLogFile.write('## number of hidden layers : {:d} \n'.format(config.num_layers))

            print('## number of hidden units : ', config.n_hidden)
            trainingLogFile.write('## number of hidden units : {:d} \n'.format(config.n_hidden))

            print('## learning rate : ', config.learning_rate)
            trainingLogFile.write('## learning rate : {:.6f} \n'.format(config.learning_rate))

            print('## batch size : ', config.batch_size)
            trainingLogFile.write('## batch size : {:d} \n'.format(config.batch_size))

            step = 1
            accuracy_list=[]

            for epoch in range(1,config.num_epochs):
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
                        _, C, train_pred, train_label , lr = sess.run([train_model.optimize, train_model.cost,
                                                                  train_model.prediction,
                                                                  train_model.labels,
                                                                  train_model.learning_rate])

                        epoch_cost += C

                        if (step % 10 == 0 or step == 1):
                            print("step[{:7d}] cost[{:2.5f}] lr[{:.6f}] time[{}]".format(step, C,lr,time.time()-partial_time))
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
                            example_accuracy, val_label, val_prediction, attention = sess.run([val_model.accuracy,
                                                                                    val_model.labels,
                                                                                    val_model.prediction,
                                                                                    val_model.attention])
                            accuracy += example_accuracy
                            if val_example_number==0:
                                print('label[{}] prediction[{}] accuracy[{}] length[{}]'.format(val_label,
                                                                                     val_prediction,
                                                                                     example_accuracy,
                                                                                     attention.shape[1]))
                                print('attention:')
                                print(attention[0,:,:])
                            val_example_number+=1

                        except tf.errors.OutOfRangeError:
                            break
                    accuracy /= num_examples_val
                    print('Validation accuracy : {} '.format(accuracy))
                    accuracy_list.append(accuracy)
                    if len(accuracy_list) > 5:
                        if accuracy <= (max(accuracy_list[-5]) + 0.0001):
                            config.learning_rate /= 2.0
                    	

                    save_path = saver.save(sess, checkpoints_dir + 'model_epoch' + str(epoch) + '.ckpt')
                    print("model saved in file: %s" % save_path)

                    outstring='{}\t{}\t{}\n'.format(epoch,epoch_cost,accuracy)
                    trainingLogFile.write(outstring)
                    trainingLogFile.flush()


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
