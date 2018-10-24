import numpy as np
import pandas as pd
import glob

from python_speech_features import delta as get_delta
from python_speech_features import fbank as get_fbanks
import scipy.io.wavfile as wav
import tensorflow as tf

kaggle_path='/home/local/IIT/rtavarone/KaggleFreeSoundChallenge/Data/'
wav_path = kaggle_path+'audio_test/'

def serialize_sequence(audio_sequence):
	# The object we return
	ex = tf.train.SequenceExample()
	# A non-sequential feature of our example
	sequence_length = len(audio_sequence)
	ex.context.feature["length"].int64_list.value.append(sequence_length)

	# Feature lists for the two sequential features of our example
	fl_audio_feat = ex.feature_lists.feature_list["audio_feat"]

	for audio_feat in audio_sequence:
		fl_audio_feat.feature.add().float_list.value.extend(audio_feat)

	return ex


wav_files=sorted(glob.glob(wav_path+'*.wav'))
wav_files.remove(wav_path+'0b0427e2.wav')
wav_files.remove(wav_path+'6ea0099f.wav')
wav_files.remove(wav_path+'b39975f5.wav')

#print(wav_files)

for file in wav_files:
	print('file = ', file)
	(rate, sig) = wav.read(file)
	fbanks, energy = get_fbanks(sig, rate, nfilt=40, nfft=2048)
	energy = np.expand_dims(energy, axis=1)
	fbanks = np.concatenate((fbanks, energy), axis=1)

	print('rate = ', rate)
	print('fbanks shape    = ', fbanks.shape)
	print('energy shape    = ', energy.shape)

	features = fbanks

	mean = np.mean(features, axis=0)
	stdev = np.std(features, axis=0)
	features = np.subtract(features, mean) / stdev


	filename = file.replace(wav_path,'').replace('.wav','')
	filename = 'test_feat/'+filename+'.tfrecords'

	fp = open(filename, 'w')
	writer = tf.python_io.TFRecordWriter(fp.name)
	serialized_sentence = serialize_sequence(features)

	writer.write(serialized_sentence.SerializeToString())
	writer.close()
	fp.close()



