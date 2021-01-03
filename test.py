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
from Face_encoder import model
from collections import OrderedDict
from PIL import Image


def load_image(image_dir):
    image = Image.open(image_dir).convert('RGB')
    if image.size != (224, 224):
        image = image.resize((224, 224), resample=Image.BILINEAR)
    image = np.array(image, dtype=np.float32)
    print('img', image)
    image = image * 2 / 255 - 1
    # # crop & resize
    # image = np.array(image,dtype=np.float32)
    image = image.transpose(2, 0, 1)
    # image = image * 2 / 255 -1
    print(image.shape)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    print(image.shape)
    return image

def get_mel_image(mel, imagetitle='Mel-Spectrogram', filename='melspec.png'):
    fig = plt.figure()
    plt.imshow(mel.squeeze().cpu().detach().numpy())
    plt.gca().invert_yaxis()
    #plt.colorbar()
    plt.title(imagetitle)
    plt.xlabel('encoder timestep')
    plt.ylabel('decoder timestep')
    fig.savefig(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="E:/FR-PSS/encoder/encoder.pt",
                        help="encoder weight")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="E:/FR-PSS/synthesizer/logs-pretrained/",
                        help="synthesizer weight")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="E:/FR-PSS/vocoder/vocoder.pt",
                        help="vocoder weight")
    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)

    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)
    face_model = model.InceptionResnetV1(dropout_prob=0.6)
    face_checkpoint = torch.load('E:/FR-PSS/Face_encoder/weight/weight_epoch_40.pth', map_location='cpu')
    new_parameter_check = OrderedDict()
    for key,value in face_checkpoint.items():
        if key.startswith('module.'):
            #print('22222222222', key[7:])
            new_parameter_check[key[7:]] = value
        #print(new_parameter_check.keys())
    model_dict = face_model.state_dict()
    pretrained_dict = {k: v for k, v in new_parameter_check.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    face_model.load_state_dict(model_dict)
    #face_model.load_state_dict(new_parameter_check)
    for para in face_model.parameters():
        para.requires_grad = False
    face_model.eval()



    #in_face_path = 'D:/BaiduNetdiskDownload/FV_dataset/FACE/Abigail_Spencer/00000049.jpg'
    #in_face_path = r'D:\BaiduNetdiskDownload\FV_dataset\FACE\Carrie_Fisher\00000121.jpg'
    in_face_path = r'D:\BaiduNetdiskDownload\FV_dataset\FACE\Rob_Brydon\00000006.jpg'
    #in_face_path = 'D:/BaiduNetdiskDownload/FV_dataset/FACE/Adam_Beach/00000098.jpg'
    #in_face_path = r'D:\BaiduNetdiskDownload\FV_dataset\FACE\Aidan_Turner\00000088.jpg'
    #in_face_path = r'D:\BaiduNetdiskDownload\FV_dataset\FACE\Max_Adler\00000025.jpg'
    #in_face_path = r'D:\BaiduNetdiskDownload\FV_dataset\FACE\Michael_Mando\00000602.jpg'
    face_load = load_image(in_face_path)

    neutral = torch.from_numpy(np.load('E:/FR-PSS/Face_encoder/neutral.npy'))
    minib = face_load.size(0)
    mean = torch.zeros(minib,256)
    for i in range(minib):
        mean[i,:] = neutral
    face_embedding = face_model(face_load,mean=mean,fuse=True)

    #face_embedding = face_model(face_load)
    print(face_embedding)
    face_embedding = face_embedding.squeeze(0)
    embed = face_embedding.numpy()

    print("Loaded file succesfully")
    print(embed.shape)
    print(type(embed))
    print("Created the embedding")

    text = 'Hope is a good thing and maybe the best of things. And no good thing ever dies.'

    texts = [text]
    embeds = [embed]

    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]

    get_mel_image(torch.from_numpy(np.array(specs)), filename='mel_wav.png', imagetitle='mel_wav')

    ## Generating the waveform

    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")


    fpath = "demo_output_%02d.wav" % 1
    print(generated_wav.dtype)
    librosa.output.write_wav(fpath, generated_wav.astype(np.float32),synthesizer.sample_rate)
    print("\nSaved output as %s\n\n" % fpath)