from encoder.audio import preprocess_wav
from encoder import inference as encoder
from pathlib import Path
import numpy as np
import librosa
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from Face_encoder import model
from collections import OrderedDict
import torch
import torch.nn as nn

enc_model_fpath = Path("C:/Users/hxs/Desktop/pretrained/encoder/saved_models/pretrained.pt")
encoder.load_model(enc_model_fpath)

class Data_ganerator(Dataset):
    def __init__(self,txt_file):
        with open(txt_file, 'r') as f:
            self.all_triplets = f.readlines()
        self.all_triplets = self.all_triplets[0:]
        print(self.all_triplets)


    def __getitem__(self, item):
        triplet = self.all_triplets[item].split(' ')
        #pos_image_dir = triplet[0]
        voice_dir = triplet[0]
        # print('1',image_dir)
        # print('2',voice_dir)
        #pos_image = self.load_image(pos_image_dir)
        voice = self.load_voice(voice_dir)
        return voice


    def __len__(self):

        return len(self.all_triplets)


    def load_voice(sele,voice_dir):
        original_wav, sampling_rate = librosa.load(voice_dir)
        preprocessed_wav = preprocess_wav(original_wav, sampling_rate)
        voice_embed = encoder.embed_utterance(preprocessed_wav)
        return voice_embed

if __name__ == '__main__':
    dataset = Data_ganerator('D:/BaiduNetdiskDownload/FV_dataset/calculfeature.txt') #中性
    loader = DataLoader(dataset,batch_size=1,shuffle=True,drop_last=True,num_workers=2)
    voice_neutral = torch.zeros((256))
    length = len(dataset)
    for step, (voice) in enumerate(loader):
        #print(pos_face.shape)
        print('voice',voice.shape)
        sumvoice = torch.sum(voice, dim=0)
        voice_neutral += sumvoice
        print(step)
    mean_voice = voice_neutral / length
    mean_voice = mean_voice.unsqueeze(dim=0)
    print(mean_voice.shape)
    mean = mean_voice.data.cpu().numpy()
    # np.save("/home/zyc/PycharmProjects/s2f/meanface.npy", mean)
    np.save("D:/BaiduNetdiskDownload/FV_dataset/10000.npy", mean)
