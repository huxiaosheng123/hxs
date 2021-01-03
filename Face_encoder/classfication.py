import  keras
from keras.models import load_model
import cv2
import numpy as np
CASE_PATH = "E:/PycharmProjects/Age-Gender-Test-master/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASE_PATH)
face_recognition_model = keras.Sequential()
MODEL_PATH = 'file_model.h5'
face_recognition_model = load_model(MODEL_PATH)

def resize_without_deformation(image, size = (32, 32)):
    height, width,_ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left
    image_with_border = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])
    resized_image = cv2.resize(image_with_border, size)
    return resized_image

def getGenderForecast(img):
    #加载卷积神经网络模型：
    IMAGE_SIZE = 32
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.2,
                                          minNeighbors=2,
                                          minSize=(
                                          2, 2), )
    for (x, y, width, height) in faces:
        img = gray[y:y + height, x:x + width]
        img = cv2.resize(img,(32,32))
        img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
        img = np.asarray(img, dtype=np.float32)
        img /= 255.0
        result = face_recognition_model.predict_classes(img)
    return result

def getFace(img):
    #加载卷积神经网络模型：
    IMAGE_SIZE = 32
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.2,
                                          minNeighbors=2,
                                          minSize=(
                                          2, 2), )
    for (x, y, width, height) in faces:
        img = img[y:y + height, x:x + width]
        img=cv2.resize(img,(250,250))
    return img
if __name__ == '__main__':
    #男的为1女的为0
    img = cv2.imread('D:/BaiduNetdiskDownload/TEST2/3/face.jpg')
    result=getGenderForecast(img)
    #print(type(result))
    print(result)
    # if result==0:
    #     print("女")
    # else:
    #     print("男")
