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
                        default="C:/Users/hxs/Desktop/pretrained/encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="C:/Users/hxs/Desktop/pretrained/synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="C:/Users/hxs/Desktop/pretrained/vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
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
    face_embedding = torch.from_numpy(np.load("E:/PycharmProjects/Real-Time-Voice-Cloning/neutral.npy"))
    print(face_embedding)
    face_embedding = face_embedding.squeeze(0)
    embed = face_embedding.numpy()

    print("Loaded file succesfully")
    print(embed.shape)
    print(type(embed))
    print("Created the embedding")

    text = 'Hope is a good thing and maybe the best of things. And no good thing ever dies.'
    print('ttttttt',text)

    texts = [text]
    embeds = [embed]
    print(type(embed))
    print(embed.shape)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print(spec.shape)
    print('spec',spec)
    get_mel_image(torch.from_numpy(np.array(specs)), filename='mel_wav.png', imagetitle='mel_wav')
    print("Created the mel spectrogram")

    ## Generating the waveform
    print("Synthesizing the waveform:")
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")


    fpath = "E:/PycharmProjects/Real-Time-Voice-Cloning/res/demo_output_%02d.wav" % 1
    print(generated_wav.dtype)
    librosa.output.write_wav(fpath, generated_wav.astype(np.float32),synthesizer.sample_rate)
    print("\nSaved output as %s\n\n" % fpath)