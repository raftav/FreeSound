import glob
import random

def get_training_files(epoch):
    random.seed(epoch)
    file_list=[]
    max_examples_per_class=85
    #max_examples_per_class=1

    for label_class in range(0,41):
        class_files=sorted(glob.glob('/home/rtavaron/KaggleFreesound/data/features_train/sequence_class{:d}_*'.format(label_class)))
        rand_smpl = [class_files[i] for i in sorted(random.sample(range(len(class_files)), max_examples_per_class))]

        for file in rand_smpl:
            file_list.append(file)
    random.shuffle(file_list)
    return file_list

def get_dev_files():
    files=sorted(glob.glob('/home/rtavaron/KaggleFreesound/data/features_dev/*.tfrecords'))
    return files



