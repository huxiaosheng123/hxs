import torch
import os
import numpy as np
import librosa
import random
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from encoder.audio import preprocess_wav
from encoder import inference as encoder
from pathlib import Path

enc_model_fpath = Path("E:/FR-PSS/encoder/encoder.pt")
encoder.load_model(enc_model_fpath)

class Data_ganerator(Dataset):
    def __init__(self,txt_file):
        with open(txt_file, 'r') as f:
            self.all_triplets = f.readlines()
        self.all_triplets = self.all_triplets[0:]
        print(self.all_triplets)


    def __getitem__(self, item):
        triplet = self.all_triplets[item].split(' ')
        pos_image_dir = triplet[1]
        neg_image_dir = triplet[2]
        voice_dir = triplet[0]
        # print('1',image_dir)
        # print('2',voice_dir)
        pos_image = self.load_image(pos_image_dir)
        neg_image = self.load_image(neg_image_dir)
        voice = self.load_voice(voice_dir)
        return voice,pos_image,neg_image


    def __len__(self):

        return len(self.all_triplets)




    def load_image(self,image_dir):
        image = Image.open(image_dir).convert('RGB')
        if image.size != (224, 224):
            image = image.resize((224, 224), resample=Image.BILINEAR)
        image = np.array(image, dtype=np.float32)
        image = image * 2 / 255 -1
        # # crop & resize
        # image = np.array(image,dtype=np.float32)
        image = image.transpose(2,0,1)
        # image = image * 2 / 255 -1
        image = torch.from_numpy(image)
        return image

    def load_voice(sele,voice_dir):
        original_wav, sampling_rate = librosa.load(voice_dir)
        preprocessed_wav = preprocess_wav(original_wav, sampling_rate)
        voice_embed = encoder.embed_utterance(preprocessed_wav)
        return voice_embed

# def custom_collate_fn(batch):
# #     voice = [torch.from_numpy(item[0]) for item in batch]
# #     face = [item[1] for item in batch]
# #     # face = torch.from_numpy(np.array(face,dtype=np.float64))
# #     # voice = torch.from_numpy(np.array(voice,dtype=np.float64))
# #     # print(face.shape)
# #     # print(voice.shape)
# #     voice = torch.stack(voice, dim=0)
# #     # face = torch.stack(face, dim=0)
# #
# #     return voice,face

# if __name__ == '__main__':
#     dataset = Data_ganerator('D:/BaiduNetdiskDownload/FV_dataset/train.txt')
#     loader = DataLoader(dataset,batch_size=2,shuffle=True,drop_last=True,num_workers=2)
#     for step, (voice,pos_face,neg_face) in enumerate(loader):
#         #print(voice)
#         print(pos_face.shape)
#         print(neg_face.shape)
#         print(voice.shape)
