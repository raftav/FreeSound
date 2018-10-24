import glob
import random
import tensorflow as tf

def _parser(example_proto):
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(example_proto,
                                                                       context_features={
                                                                           "length": tf.FixedLenFeature([],
                                                                                                        dtype=tf.int64)},
                                                                       sequence_features={
                                                                           "audio_feat": tf.FixedLenSequenceFeature(
                                                                               [41], dtype=tf.float32),
                                                                           "audio_labels": tf.FixedLenSequenceFeature(
                                                                               [], dtype=tf.float32)}
                                                                       )
    return context_parsed['length'], sequence_parsed['audio_feat'], tf.to_int32(sequence_parsed['audio_labels'])

def _test_parser(example_proto):
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(example_proto,
                                                                       context_features={
                                                                           "length": tf.FixedLenFeature([],
                                                                                                        dtype=tf.int64)},
                                                                       sequence_features={
                                                                           "audio_feat": tf.FixedLenSequenceFeature(
                                                                               [41], dtype=tf.float32)}
                                                                       )
    return context_parsed['length'], sequence_parsed['audio_feat']

def get_training_files(epoch):
    random.seed(epoch)
    file_list=[]
    max_examples_per_class=564
    #max_examples_per_class=1

    for label_class in range(0,41):
        class_files=sorted(glob.glob('/home/rtavaron/KaggleFreesound/data/train_feat/sequence_class{:d}_*'.format(label_class)))
        rand_smpl = [class_files[i] for i in sorted(random.sample(range(len(class_files)), max_examples_per_class))]

        for file in rand_smpl:
            file_list.append(file)
    random.shuffle(file_list)
    return file_list

def get_dev_files():
    files=sorted(glob.glob('/home/rtavaron/KaggleFreesound/data/val_feat/*.tfrecords'))
    return files


def get_test_files():
    files=sorted(glob.glob('/home/rtavaron/KaggleFreesound/data/test_feat/*.tfrecords'))
    return files
