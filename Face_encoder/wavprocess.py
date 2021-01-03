import os
import shutil
meta_data = r'D:\BaiduNetdiskDownload\dataset\VoxCeleb1\vox1_meta.csv'

# def loadTxtData(txt_path):
#     f=open(txt_path,'r')
#     sourceInLine=f.readlines()
#     sourceInLine = sourceInLine[1:]
#     for line in sourceInLine:
#         temp=str(line).strip('\n').split('\t')[:2]
#         #print(temp)
#         voxceleb = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/voxceleb'+'/'+'VoxCeleb1_wav_'+temp[0]
#         vggface = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/VGG_ALL_FRONTAL'+ '/' +temp[1]
#         new_voxceleb = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/WAV' + '/' + temp[1]
#         new_vggface = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/FACE' + '/' + temp[1]
#         if not os.path.exists(new_voxceleb):
#             os.mkdir(new_voxceleb)
#         if not os.path.exists(new_vggface):
#             os.mkdir(new_vggface)
#         print(voxceleb)
#         print(vggface)
#         try:
#             for voicefle in os.listdir(voxceleb):
#                 print(voicefle)
#                 if voicefle.endswith('npy'):
#                     voicefle_path = voxceleb + '/' + voicefle
#                     print(voicefle_path)
#                     print(new_voxceleb)
#                     shutil.copy(voicefle_path,new_voxceleb)
#         except:
#             continue
#         try:
#             for facefile in os.listdir(vggface):
#                 face_path = vggface + '/' + facefile
#                 print(face_path)
#                 shutil.copy(face_path, new_vggface)
#         except:
#             continue
#
# loadTxtData(meta_data)


# ##判断文件夹是否为空
# # 导入os
# import os
# # 让用户自行输入路径
# path='D:/BaiduNetdiskDownload/dataset/SV2TTS/WAV'
# # 获取当前目录下的所有文件夹名称  得到的是一个列表
#
# def folder_is_None(path):
#     folders = os.listdir(path)
#     print(folders)
#     for folder in folders:
#         # 将上级路径path与文件夹名称folder拼接出文件夹的路径
#         folder2 = os.listdir(path + '/' + folder)
#         print(folder2)
#         if folder2 == []:
#             # 则打印此空文件的名称
#             print(folder)
#             with open('D:/BaiduNetdiskDownload/dataset/SV2TTS/wav_none.txt', "a") as file:
#                 file.write(folder+'\n')
#             # 并将此空文件夹删除
#             os.rmdir(path + '\\' + folder)
#
# folder_is_None(path)

# #删除与voxceleb不对应的face文件夹
# path = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/wav_none.txt'
#
# def delete_face_folder(txt_path):
#     dataset = []
#     with open(txt_path) as txt:
#         name = txt.readlines()
#         for linename in name:
#             temp = linename.strip('\n')
#             dataset.append(temp)
#         print(dataset)
#         for linename1 in dataset:
#             path = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/FACE'+'/'+ linename1
#             print(path)
#             # # print(len(path))
#             os.rmdir(path)
#
# delete_face_folder(path)

##判断FACe里是否存在空文件夹
import os
# 让用户自行输入路径
#path='D:/BaiduNetdiskDownload/dataset/SV2TTS/FACE'
# 获取当前目录下的所有文件夹名称  得到的是一个列表

# def folder_is_None(path):
#     folders = os.listdir(path)
#     print(folders)
#     for folder in folders:
#         # 将上级路径path与文件夹名称folder拼接出文件夹的路径
#         folder2 = os.listdir(path + '/' + folder)
#         print(folder2)
#         if folder2 == []:
#             # 则打印此空文件的名称
#             print(folder)
#             with open('D:/BaiduNetdiskDownload/dataset/SV2TTS/face_none.txt', "a") as file:
#                 file.write(folder+'\n')
#             os.rmdir(path + '\\' + folder)
#
# folder_is_None(path)

# 删除与face不对应的fwav文件夹
# import shutil
# path = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/face_none.txt'
#
# def delete_face_folder(txt_path):
#     dataset = []
#     with open(txt_path) as txt:
#         name = txt.readlines()
#         for linename in name:
#             temp = linename.strip('\n')
#             dataset.append(temp)
#         print(dataset)
#         for linename1 in dataset:
#             path = 'D:/BaiduNetdiskDownload/dataset/SV2TTS/WAV'+'/'+ linename1
#             print(path)
#             # # print(len(path))
#             shutil.rmtree(path)
#
# delete_face_folder(path)

# ##整理voxceleb的wav文件夹
# import os
#
# wavpath = 'D:/BaiduNetdiskDownload/dataset/VoxCeleb1/wav'
# txtpath = 'D:/BaiduNetdiskDownload/dataset/VoxCeleb1/vox1_meta.csv'
# txtdataset= []
# def load_wav(txtpath):
#     f = open(txtpath,'r')
#     lines = f.readlines()
#     for line in lines[1:]:
#         line1 = line.strip('\n').split('\t')[:2]
#         #print(line1) #['id11250', 'Zoe_Saldana']
#         wav_path = wavpath + '/' +line1[0]
#         #print(wav_path) #D:/BaiduNetdiskDownload/dataset/VoxCeleb1/wav/id11250
#         wav_newname = 'D:/BaiduNetdiskDownload/dataset/VoxCeleb11' + '/' + line1[1] #D:/BaiduNetdiskDownload/dataset/VoxCeleb11/Zoe_Saldana
#         if not os.path.exists(wav_newname):
#             os.mkdir(wav_newname)
#         print(wav_newname)
#         i = 1
#         for filewav in os.listdir(wav_path):
#             path1 = wav_path + '/' + filewav
#             print(path1) #D:/BaiduNetdiskDownload/dataset/VoxCeleb1/wav/id10001/J9lHsKG98U8
#             for filewav1 in os.listdir(path1):
#                 src = path1 + '/'+filewav1
#                 print('111111111',src)
#                 target = wav_newname + '/' + '{}.wav'.format(i)
#                 print('222222222',target)
#                 i = i+1
#                 shutil.copyfile(src,target)
#
#
# load_wav(txtpath)


#import shutil
# path = 'D:/BaiduNetdiskDownload/dataset_encoder/SV2TTS1/WAV'
#
# def delete_wav_folder(voice_path):
#     dataset = []
#     for filename in os.listdir(voice_path):
#         print(filename)
#         with open('D:/BaiduNetdiskDownload/dataset/SV2TTS/vox.txt', "a") as file:
#             file.write(filename+'\n')
#
# delete_wav_folder(path)

# wav_path = 'D:/BaiduNetdiskDownload/dataset/VoxCeleb11'
# txt_path = 'D:/BaiduNetdiskDownload/dataset/vox.txt'
# def delete_wav_folder(txt_path,voice_path):
#     dataset = []
#     with open(txt_path) as txt:
#             name = txt.readlines()
#             for linename in name:
#                 temp = linename.strip('\n')
#                 dataset.append(temp)
#             print(dataset)
#
#     for filename in os.listdir(voice_path):
#         print(filename)
#         if filename not in dataset:
#             with open('D:/BaiduNetdiskDownload/dataset/SV2TTS/vox_delet.txt', "a") as file:
#                 file.write(filename+'\n')

# delete_wav_folder(txt_path,wav_path)

#删除与face不对应的fwav文件夹
import shutil
path = 'D:/BaiduNetdiskDownload/dataset/vox_delet.txt'

def delete_face_folder(txt_path):
    dataset = []
    with open(txt_path) as txt:
        name = txt.readlines()
        for linename in name:
            temp = linename.strip('\n')
            dataset.append(temp)
        print(dataset)
        print(len(dataset))
        for linename1 in dataset:
            path = 'D:/BaiduNetdiskDownload/dataset/VoxCeleb11'+'/'+ linename1
            print(path)
            # # print(len(path))
            shutil.rmtree(path)

delete_face_folder(path)