import os
import random
from sklearn.utils import shuffle

voice_path = 'D:/BaiduNetdiskDownload/FV_dataset/VoxCeleb11'
face_path = 'D:/BaiduNetdiskDownload/FV_dataset/FACE'

def get_file_path_len(name):
    path = voice_path + '/' + name
    #print('22222222222',path)
    dataset = []
    for name in os.listdir(path):
        #print(name)
        dataset.append(name)
    return dataset

def get_imgfile_path_len(name):
    path = face_path + '/' + name
    #print('22222222222',path)
    dataset = []
    for name in os.listdir(path):
        #print(name)
        dataset.append(name)
    return dataset

name_file = os.listdir(voice_path)
print(name_file)
len_name_file = len(name_file)
dataset_list = []
for name in name_file:
    print(name)
    wav_path = voice_path + '/' + name
    pos_img_path = face_path + '/' +name
    # neg_name = name_file[random.randint(0, len_name_file-1)]
    # while neg_name==name:
    #     neg_name = name_file[random.randint(0, len_name_file - 1)]
    # neg_img_path = face_path + '/' + neg_name
    print(wav_path)
    print(pos_img_path)
    #print(neg_img_path)
    wav_data = get_file_path_len(name)
    wav_len = len(wav_data)
    pos_img_data = get_imgfile_path_len(name)
    pos_img_data_len = len(pos_img_data)
    #print(wav_len)
    for i in range(wav_len-1):
        wav_path1 = wav_path + '/' + wav_data[i]
        pos_img_path1 = pos_img_path + '/' + pos_img_data[random.randint(0,pos_img_data_len-1)]
        neg_name = name_file[random.randint(0, len_name_file - 1)]
        while neg_name == name:
            neg_name = name_file[random.randint(0, len_name_file - 1)]
        neg_img_path = face_path + '/' + neg_name
        print(neg_img_path)
        neg_img_data = get_imgfile_path_len(neg_name)
        neg_img_data_len = len(neg_img_data)
        neg_img_path1 = neg_img_path + '/' + neg_img_data[random.randint(0,neg_img_data_len-1)]
        print('*',wav_path1)
        print('*',pos_img_path1)
        print('*',neg_img_path1)
        a = [wav_path1,pos_img_path1,neg_img_path1]
        dataset_list.append(a)
print('list',dataset_list)
total_len = len(dataset_list)
eight_part = int(total_len*0.8)
all_shuffle = shuffle(dataset_list)
train_dataset = all_shuffle[:eight_part]
val_dataset = all_shuffle[eight_part:]
for train_data in train_dataset:
    print(train_data)
    train_triple = ' '.join((train_data[0], train_data[1], train_data[2], ''))
    file = open("D:/BaiduNetdiskDownload/FV_dataset/train.txt", 'a')
    file.write(train_triple+'\n')

for val_data in val_dataset:
    val_triple = ' '.join((val_data[0], val_data[1], val_data[2], ''))
    file = open("D:/BaiduNetdiskDownload/FV_dataset/val.txt", 'a')
    file.write(val_triple+'\n')









# for name in name_file:
#     print('1111111111',name)
#     data = get_file_path_len(name)
#     len = len(data)
#     for _ in range(len):
#         get_triplet(name)
#     print(name)




