import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sys
from tensorflow.python.client import timeline

import lstm_model
from utils import input_pipeline


#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])


class Configuration(object):
    learning_rate = float(sys.argv[2])
    batch_size = int(sys.argv[3])
    num_layers = int(sys.argv[4])
    n_hidden = int(sys.argv[5])
    
    audio_feat_dimension = 41
    audio_labels_dim = 41
    num_epochs = 5000
    keep_prob=0.8
    num_gpus=16

    masknet_num_neurons=50
    
    filter_size=5
    filter_out_channels=audio_feat_dimension

checkpoints_dir = 'checkpoints/exp' + str(ExpNum) + '/'

tensorboard_dir = 'tensorboard/exp' + str(ExpNum) + '/'

trainingLogFile = open('TrainingExperiment' + str(ExpNum) + '.txt', 'w')

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

# Build the function to average the gradients
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


#################################
# Training module
#################################
def train():
    config = Configuration()

    # training graph
    with tf.Graph().as_default():

        # extract batch examples
        with tf.device('/cpu:0'):
            filenames = tf.placeholder(tf.string, shape=[None])
            batch_size = tf.placeholder(tf.int64, shape=())

            dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=32,buffer_size=100)
            dataset = dataset.map(input_pipeline._parser, num_parallel_calls=64)
            dataset = dataset.repeat(1)

            # dataset = dataset.batch(batch_size)
            dataset = dataset.padded_batch(batch_size, padded_shapes=([],
                                                                      [None, 41],
                                                                      [None])).filter(lambda x, y, z: tf.equal(tf.shape(x)[0], tf.cast(batch_size,tf.int32)))
            
            dataset = dataset.prefetch(50)

            iterator = dataset.make_initializable_iterator()

            sequence_length, features, labels = iterator.get_next()

        # place tower gradients on cpu
        with tf.device('/cpu:0'):
            tower_grads = []
            tower_loss = []
            global_step = tf.Variable(0, trainable=False)
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

        # split batched data between gpus
        split_seq_len = tf.split(sequence_length,num_or_size_splits=config.num_gpus,axis=0)
        split_features = tf.split(features,num_or_size_splits=config.num_gpus,axis=0)
        split_labels = tf.split(labels,num_or_size_splits=config.num_gpus,axis=0)

        reuse_vars = False
        for i in range(config.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.variable_scope('model',reuse=reuse_vars):
                    model=lstm_model.Model(split_features[i],split_seq_len[i],config,is_training=True)
                    logits=model.logits

                    with tf.name_scope('cost'):
                        cost = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=split_labels[i]))
                        tower_loss.append(cost)

                    # gradient clipping
                    gradients, variables = zip(*optimizer.compute_gradients(cost))
                    clip_grad = [None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in gradients]
                    clip_grads_and_vars = zip(clip_grad, variables)
                        
                    tower_grads.append(clip_grads_and_vars)
                    reuse_vars=True
        
        tower_grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(tower_grads,global_step=global_step)
        loss = tf.reduce_mean(tower_loss)

        with tf.device('/cpu:0'):
            with tf.variable_scope('model',reuse=True):
                print('Building val model:')
                val_model = lstm_model.Model(features, sequence_length, config,is_training=False)
                print('done.\n')
            val_logits=val_model.logits
            posteriors = tf.nn.softmax(val_logits)
            prediction = tf.argmax(val_logits, axis=2)
            correct = tf.equal(prediction, tf.to_int64(labels))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # variables initializer
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=10)
        #run_metadata = tf.RunMetadata()

        # start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:

            sess.run(init_op)

            #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            
            print('')
            print('## EXPERIMENT NUMBER ', ExpNum)
            trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))

            print('## number of hidden layers : ', config.num_layers)
            trainingLogFile.write('## number of hidden layers : {:d} \n'.format(config.num_layers))

            print('## number of hidden units : ', config.n_hidden)
            trainingLogFile.write('## number of hidden units : {:d} \n'.format(config.n_hidden))

            print('## learning rate : ', config.learning_rate)
            trainingLogFile.write('## learning rate : {:.6f} \n'.format(config.learning_rate))

            print('## batch size : ', config.batch_size)
            trainingLogFile.write('## batch size : {:d} \n'.format(config.batch_size))

            
            accuracy_list=[]

            for epoch in range(1,config.num_epochs):
                step = 1
                filenames_train = input_pipeline.get_training_files(epoch)
                num_examples = len(filenames_train)
                print('')
                print('Starting epoch ',epoch)
                print('Num training examples = ',num_examples)

                sess.run(iterator.initializer, feed_dict={filenames: filenames_train,
                                                      batch_size: config.batch_size})

                EpochStartTime = time.time()
                partial_time=time.time()

                epoch_cost = 0.0

                while True:
                    try:
                        _, C  = sess.run([train_op,loss])


                        epoch_cost += C

                        if (step % 10 == 0 or step == 1):
                            print("step[{:7d}] cost[{:2.5f}] lr[{}] time[{}]".format(step, C,config.learning_rate,time.time()-partial_time))
                            partial_time=time.time()


                        step += 1

                    except tf.errors.OutOfRangeError:
                        break

                # End-of-training-epoch calculation here
                epoch_cost /= step

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
                                                              batch_size: 10})
                    val_accuracy = 0
                    val_example_number=0

                    while True:
                        try:
                            example_accuracy, val_label, val_prediction = sess.run([accuracy,
                                                                                    labels,
                                                                                    prediction])
                            val_accuracy += example_accuracy
                            '''
                            if val_example_number==0:
                                print('label[{}] prediction[{}] accuracy[{}]'.format(val_label,
                                                                                     val_prediction,
                                                                                     example_accuracy))
                                
                                print('attention shape :',attention.shape)
                                for value in range(attention.shape[1]):
                                    print('{:.5f}'.format(attention[0,value,0]))
                                print('')
                                print('attention sum = ',np.sum(attention,axis=1))
                            '''
                            
                            val_example_number+=1

                        except tf.errors.OutOfRangeError:
                            break

                    val_accuracy /= val_example_number
                    print('Validation accuracy : {} '.format(val_accuracy))
                    print('Validation time : {}'.format(time.time()-ValidationStartTime))

                    accuracy_list.append(val_accuracy)
                    if len(accuracy_list) > 5:
                        if (val_accuracy <= (max(accuracy_list[:-5]) + 0.0001) and config.learning_rate > 0.0001):
                            config.learning_rate /= 2.0
					

                    save_path = saver.save(sess, checkpoints_dir + 'model_epoch' + str(epoch) + '.ckpt')
                    print("model saved in file: %s" % save_path)

                    outstring='{}\t{}\t{}\t{}\n'.format(epoch,config.learning_rate,epoch_cost,val_accuracy)
                    trainingLogFile.write(outstring)
                    trainingLogFile.flush()

                    sys.exit()


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
