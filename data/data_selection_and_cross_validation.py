import numpy as np
import librosa as lr
import pandas as pd
from python_speech_features import delta as get_delta
from python_speech_features import fbank as get_fbanks
import tensorflow as tf
import shutil

kaggle_path='/home/local/IIT/rtavarone/KaggleFreeSoundChallenge/Data/'
train_files = pd.read_csv(kaggle_path+'train.csv', header=0)
labels_set = pd.read_csv(kaggle_path+'labels.csv', header=0)
wav_path = kaggle_path+'audio_train/'

def serialize_sequence(audio_sequence, labels):
	# The object we return
	ex = tf.train.SequenceExample()
	# A non-sequential feature of our example
	sequence_length = len(audio_sequence)
	ex.context.feature["length"].int64_list.value.append(sequence_length)

	# Feature lists for the two sequential features of our example
	fl_audio_feat = ex.feature_lists.feature_list["audio_feat"]
	fl_audio_labels = ex.feature_lists.feature_list["audio_labels"]
	fl_audio_labels.feature.add().float_list.value.append(label)

	for audio_feat in audio_sequence:
		fl_audio_feat.feature.add().float_list.value.extend(audio_feat)

	return ex

def feature_extractor(data,rate,label,filename):
	fbanks, energy = get_fbanks(data, rate, nfilt=40, nfft=2048)
	energy = np.expand_dims(energy, axis=1)
	fbanks = np.concatenate((fbanks, energy), axis=1)

	features = fbanks

	mean = np.mean(features, axis=0)
	stdev = np.std(features, axis=0)
	features = np.subtract(features, mean) / stdev

	fp = open(filename, 'w')
	writer = tf.python_io.TFRecordWriter(fp.name)
	serialized_sentence = serialize_sequence(features, label)

	# write to tfrecord
	writer.write(serialized_sentence.SerializeToString())
	writer.close()
	fp.close()


def add_white_noise(data,stddev=0.01):
	wn = np.random.randn(len(data))
	data_wn = data + stddev*wn
	return data_wn

def time_stretch(data, rate=1):
	data_stretch = lr.effects.time_stretch(data, rate)
	return data_stretch

def pitch_shift(data,rate,shift_steps):
	new_data = lr.effects.pitch_shift(data,rate,shift_steps)
	return new_data


train_examples_counter = np.ones(labels_set.shape[0], dtype=int)
train_files = train_files[train_files.manually_verified==1]
num_labels = []

for index, row in train_files.iterrows():
	print('{}\t Processing file {}'.format(index,row.fname))
	string_label = row.label
	print('string label = ',string_label)

	label = int(labels_set[labels_set.label == string_label].index.values[0])
	train_examples_counter[label] += 1
	num_labels.append(label)

for example_class in range(labels_set.shape[0]):
	print('class {} num examples  = {}'.format(example_class,train_examples_counter[example_class]))

min_num_examples=np.amin(train_examples_counter)
print('Minimum number of example = ',min_num_examples)

val_example_per_class = int(min_num_examples*0.2)
print('number validation example per class = ',val_example_per_class)

train_files['num_label']=num_labels
count_train_examples=[]
for num_label in range(labels_set.shape[0]):
	class_files=train_files[train_files.num_label==num_label]
	class_dev_files = class_files.sample(n=val_example_per_class)
	class_train_files = class_files.drop(class_dev_files.index)
	print('label index = {}'.format(num_label))
	print('num files = {}'.format(len(class_files)))
	print('num validation files = {}'.format(len(class_dev_files)))
	print('num training files = {}'.format(len(class_train_files)))

	print('')

	train_examples_counter=0
	for index,file in class_train_files.iterrows():
		src=kaggle_path+'audio_train/'+file.fname
		dest='/home/local/IIT/rtavarone/KaggleFreeSoundChallenge/Data_2/train_wavs/'+file.fname
		shutil.copy(src,dest)

		data , rate = lr.core.load(dest,sr=None)

		#start data augmentation
		noise_stddev=[0.0,0.005]
		time_stretch_rate=[1.0,3.0]
		pitch_shift_rate=[0.0,-3.0,3.0]
		for stddev in noise_stddev:
			for stretch in time_stretch_rate:
				for pitch in pitch_shift_rate:
					data_aug = add_white_noise(data,stddev)
					data_aug = time_stretch(data_aug,stretch)
					data_aug = pitch_shift(data_aug,rate,pitch)

					feat_filename = 'train_feat/sequence_class{:d}_{:04d}.tfrecords'.format(num_label,train_examples_counter)
					feature_extractor(data_aug,rate,num_label,feat_filename)
					train_examples_counter+=1
					
		count_train_examples.append(train_examples_counter+1)

	val_examples_counter=0
	for index,file in class_dev_files.iterrows():
		src=kaggle_path+'audio_train/'+file.fname
		dest='/home/local/IIT/rtavarone/KaggleFreeSoundChallenge/Data_2/val_wavs/'+file.fname
		shutil.copy(src,dest)

		data , rate = lr.core.load(dest,sr=None)
		feat_filename = 'val_feat/sequence_class{:d}_{:04d}.tfrecords'.format(num_label,val_examples_counter)
		feature_extractor(data,rate,num_label,feat_filename)
		val_examples_counter+=1


print('number of training examples per class:')
print(count_train_examples)
print('minumum number of training examples = ',min(count_train_examples))


