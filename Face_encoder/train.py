from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
from utils.argutils import print_args
import matplotlib.pylab as plt
from Face_encoder.dataload_FV import Data_ganerator
from torch.utils.data import DataLoader
from Face_encoder.model import InceptionResnetV1
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="E:/Real-Time-Voice-Cloning/encoder.pt",
                        help="encoder weight")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="E:/Real-Time-Voice-Cloning/synthesizer/logs-pretrained/",
                        help="synthesizer weight")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="E:/Real-Time-Voice-Cloning/vocoder.pt",
                        help="vocoder weight")

    args = parser.parse_args()
    print_args(args, parser)

    print("Preparing the encoder, the synthesizer and the vocoder...")
    #encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"))
    #vocoder.load_model(args.voc_model_fpath)

    face_model = InceptionResnetV1(pretrained=None,classify=False,num_classes=1000,dropout_prob=0.6,device=None)
    #neutral = torch.from_numpy(np.load('E:/Real-Time-Voice-Cloning/Face_encoder/neutral.npy'))
    train_data = Data_ganerator('D:/BaiduNetdiskDownload/FV_dataset/train.txt')
    train_loader = DataLoader(train_data,batch_size=2,shuffle=True,drop_last=True,num_workers=2)
    val_data = Data_ganerator('D:/BaiduNetdiskDownload/FV_dataset/val.txt')
    val_loader = DataLoader(val_data, batch_size=2, shuffle=True, drop_last=True, num_workers=2)

    Loss1 = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(face_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    weigth_save_dir = 'E:/PycharmProjects/Real-Time-Voice-Cloning/Face_encoder/weight'
    epoch_num = 500
    step_size = int(len(train_data)/2) #batch_size
    val_step_size = int(len(train_data)/2) #batch_size
    for epoch in range(1,epoch_num):
        for step, (voice,pos_face,neg_face) in enumerate(train_loader):
            # print(voice.shape) #embed
            # print(pos_face.shape)
            # print(neg_face.shape)
            #add prior
            # minib = pos_face.size(0)
            # mean = torch.zeros(minib,256)
            # for i in range(minib):
            #     mean[i,:] = neutral
            # pos_face_embed = face_model(pos_face,mean=mean,fuse=True)

            pos_face_embed = face_model(pos_face)
            neg_face_embed = face_model(neg_face)
            # print('v',voice.shape)
            # print('1111111',voice.shape)
            # print('2222222',pos_face_embed.shape)
            # print('3333333',neg_face_embed.shape)
            loss = triplet_loss(voice,pos_face_embed,neg_face_embed)+Loss1(pos_face,neg_face)
            face_model.zero_grad()
            loss.backward()
            print(loss)
            optimizer.step()
            #print("learning rate is ", optimizer.param_groups[0]["lr"])

            print('[Epoch {}/{}] [step {}/{}] LR:{:.4f} loss:{:.4f} '.format(epoch,epoch_num,step+1,step_size,
                                                                              optimizer.param_groups[0]["lr"],
                                                                                loss.item()))
        save_weight_file = 'weight_epoch_{}.pth'.format(epoch)
        print(save_weight_file)
        torch.save(face_model.state_dict(), os.path.join(weigth_save_dir, save_weight_file))

        #verification
        print('start val')
        total_val_loss = 0
        step_loss = 0
        face_model.eval()
        for step, (voice,pos_face,neg_face) in enumerate(val_loader):

            pos_face_embed = face_model(pos_face)
            neg_face_embed = face_model(neg_face)
            valloss = triplet_loss(voice,pos_face_embed,neg_face_embed)
            total_val_loss += valloss.item()
            step_loss += 1

            print('[Epoch {}/{}] [step {}/{}]  loss:{:.4f} totalloss:{:.4f}'.format(epoch, epoch_num, step + 1, val_step_size,
                                                                             valloss.item(),total_val_loss))
        totalvalloss = total_val_loss / step_loss
        with open("E:/PycharmProjects/Real-Time-Voice-Cloning/Face_encoder/log.txt", 'a') as txt:
            content = "{}".format(totalvalloss)
            txt.write(content + '\n')
