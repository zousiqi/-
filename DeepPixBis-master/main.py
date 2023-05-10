r'''
`Main` File for training DeepPix based spoof face classifier.

@Author :: `Abhishek Bhardwaj`

Labels :: `0` -> Spoof
          `1` -> Live

Pipeline Structure:: 
    Raw Image -> Crop Face -> Resize(224, 224)
         -> DeepPixNetwork -> Feature Map, Classification

Note: Feature Map is used for classification as spoof or bonafide 
        by calculating PA score

'''

import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
from models import DeepPix
from train import trainDeepPix
from imutils import paths
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report
from tkinter import *
import os

import cv2
from PIL import ImageTk
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from facenet_pytorch import MTCNN

# from faceDetector import cropFace


mtcnn = MTCNN()

# 创建窗口：实例化一个窗口对象。
root = Tk()


ji = 1


# 窗口大小
root.geometry("1000x750+374+182")

root.title("我的个性签名设计")

# 添加标签控件

label = Label(root)
# 定位
label.place(x=0, y=0, width=1000, height=700)

# cap = cv2.VideoCapture(0)  # 开启摄像头

cap = cv2.VideoCapture("http://admin:admin@192.168.43.1:8081/")

# Training on GPU `0`
torch.cuda.set_device(0)
TRAIN = False  # Set to True if you want to train
DEVICE = torch.device("cuda")
DETECT_FACE = True  # Whether to use face_detector or not

# Setting fixed seed for reproducing results
random.seed(72)
np.random.seed(72)
torch.random.manual_seed(72)

if torch.cuda.is_available():
    torch.cuda.manual_seed(72)


# Face cropping module 
mtcnn = MTCNN(margin=14, image_size=160, device=DEVICE)

# Get list of image paths
liveImagePath = []
for i in range(1, 21):
    for j in range(1, 3):
        liveImagePath += list(paths.list_images('D:/bysj/casia-fasd/train/{}/{}'.format(i, j)))
for i in range(1, 21):
    liveImagePath += list(paths.list_images('D:/bysj/casia-fasd/train/{}/HR_1'.format(i)))
spoofImagePath = []
for i in range(1, 21):
    for j in range(3, 9):
        spoofImagePath += list(paths.list_images('D:/bysj/casia-fasd/train/{}/{}'.format(i, j)))
for i in range(1, 21):
        spoofImagePath += list(paths.list_images('D:/bysj/casia-fasd/train/{}/HR_2'.format(i)))
        spoofImagePath += list(paths.list_images('D:/bysj/casia-fasd/train/{}/HR_3'.format(i)))
        spoofImagePath += list(paths.list_images('D:/bysj/casia-fasd/train/{}/HR_4'.format(i)))
# liveImagePath = list(paths.list_images('D:/毕业设计/casia-fasd/train/{}/{}'.format(1, 1)))

# spoofImagePath = list(paths.list_images('C:/Users/20946/Desktop/DeepPixBis-master/DeepPixBis-master/dataset/spoof'))

trainList = []

for live in liveImagePath:
    trainList.append([live, 1])

for spoof in spoofImagePath:
    trainList.append([spoof, 0])


# Shuffle the list to generate randomness
random.shuffle(trainList)

trainPath = [x[0] for x in trainList]
trainLabel = [x[1] for x in trainList]

# Initialize Model
model = DeepPix()

# Initialize trainer Object
trainObject = trainDeepPix(model=model, lr=1e-4, weight_decay=1e-5)

if TRAIN:

    trainObject.train(trainPath, trainLabel, batch_size=32, epochs=50, mtcnn=mtcnn, detectFace=DETECT_FACE)
    trainObject.saveModel("./adanbceloss.hdf5")

else:
    # C:\Users\20946\Desktop\DeepPixBis-master\DeepPixBis-master/DeepPixWeights.hdf5
    trainObject.loadModel("./adanbceloss.hdf5")

#Preparing Test Data

# liveImagePath = list(paths.list_images('/mnt/abhishek/Russian/val_russian_dataset_live'))
# spoofImagePath = list(paths.list_images('/mnt/abhishek/Russian/val_russian_dataset_spoof'))
liveImagePath1 = []
for i in range(1, 31):
    for j in range(1, 2):
        liveImagePath1 += list(paths.list_images('D:/bysj/casia-fasd/test/{}/{}'.format(i, j)))

spoofImagePath1 = []
for i in range(1, 31):
    for j in range(3, 9):
        spoofImagePath1 += list(paths.list_images('D:/bysj/casia-fasd/test/{}/{}'.format(i, j)))
# for i in range(1, 31):
#         spoofImagePath1 += list(paths.list_images('D:/毕业设计/casia-fasd/test/{}/HR_2'.format(i)))
#         spoofImagePath1 += list(paths.list_images('D:/毕业设计/casia-fasd/test/{}/HR_3'.format(i)))
#         spoofImagePath1 += list(paths.list_images('D:/毕业设计/casia-fasd/test/{}/HR_4'.format(i)))
# liveImagePath = list(paths.list_images('C:/Users/20946/Desktop/DeepPixBis-master/DeepPixBis-master/dataset/live'))
# spoofImagePath = list(paths.list_images('C:/Users/20946/Desktop/DeepPixBis-master/DeepPixBis-master/dataset/spoof'))

# testList = ['D:/bysj/casia-fasd/train/1/1/1 01.jpg', 1]
testList = []

for live in liveImagePath1:
    testList.append([live, 1])

for spoof in spoofImagePath1:
    testList.append([spoof, 0])

# Shuffle the list to generate randomness
random.shuffle(testList)

testPath = [x[0] for x in testList]
testLabel = [x[1] for x in testList]
# testPath = [testList[0]]
# testLabel = [testList[1]]

# Prediction from network
# pred = trainObject.predict(testPath, mtcnn=mtcnn, thresh=0.5, testLabel=testLabel, detectFace=DETECT_FACE)
# print(testLabel)
# print(pred)


def remove():
    global ji
    ji = 1
    # os.remove('D:/bysj/sxt/image.jpg')
    # im = Image.open('D:/wu.jpg')
    # im = ImageTk.PhotoImage(im)
    # label.image = im
    # label.config(image=im)
    # print("删除成功")


def imshow():
    global ji
    if ji == 0:
        im = Image.open('D:/bysj/sxt/image.jpg')
        im = im.resize((1000, 700))
        im = ImageTk.PhotoImage(im)
        label.image = im
        label.config(image=im)
    else:
        res, img = cap.read()

        if res == True:
            #将adarray转化为image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            #显示图片到label
            img = img.resize((1000, 700))
            img = ImageTk.PhotoImage(img)
            label.image = img
            # label['image'] = img
            label.config(image=img)
    #创建一个定时器，每10ms进入一次函数

    root.after(10, imshow)


def cropFace(mtcnn, PILImage, note):

    r'''
    Utility to perform face extraction from raw images
    and return a tensor of shape `channels, 224, 224`
    从原始图像中执行人脸提取的实用程序
    并返回形状通道的张量，224224`
    '''
    # name = PILImage
    PILImage = Image.open(PILImage)
    boxes, prob = mtcnn.detect(PILImage)
    print(boxes, prob)
    a = ImageDraw.Draw(PILImage)
    i = 0
    a.line(((boxes[i][0], boxes[i][1]), (boxes[i][0], boxes[i][3])), fill=(0, 255, 0), width=5)
    a.line(((boxes[i][0], boxes[i][3]), (boxes[i][2], boxes[i][3])), fill=(0, 255, 0), width=5)
    a.line(((boxes[i][2], boxes[i][3]), (boxes[i][2], boxes[i][1])), fill=(0, 255, 0), width=5)
    a.line(((boxes[i][2], boxes[i][1]), (boxes[i][0], boxes[i][1])), fill=(0, 255, 0), width=5)
    font_size = 32
    setFont = ImageFont.truetype('C:/windows/fonts/Dengl.ttf', font_size)  # 设置字体以及字体大小
    draw = ImageDraw.Draw(PILImage)  # 得到画笔
    # poem = ["静夜思", "窗前明月光", "疑是地上霜", "举头望明月", "低头思故乡"]  # 创建一首诗，等待写入
    # for i in range(len(poem)):
    print(note)
    if note == 1:
        draw.text((100, 100), "真", font=setFont, fill=(0, 255, 0))  # 利用ImageDraw的内置函数，在图片上写入文字
    else:
        draw.text((100, 100), "假", font=setFont, fill=(0, 255, 0))  # 利用ImageDraw的内置函数，在图片上写入文字
    PILImage.save("D:/bysj/sxt/image.jpg")
    im = Image.open("D:/bysj/sxt/image.jpg")
    im = im.resize((1000, 700))
    im = ImageTk.PhotoImage(im)
    label.image = im
    label.config(image=im)
    # im = ImageTk.PhotoImage(PILImage)
    # label.image = im
    # label.config(image=im)
    # label.config(text="asfqaf", fg='green')
    print("拍照成功")


def get_photo():
    global ji
    ji = 0
    global cap
    cap.release()
    cap = cv2.VideoCapture("http://admin:admin@192.168.43.1:8081/")
    f, frame = cap.read()               # 将摄像头中的一帧图片数据保存
    cv2.imwrite('D:/bysj/sxt/image.jpg', frame)     # 将图片保存为本地文件
    # D:/bysj/casia-fasd/test/14/1/1 02.jpg
    cropFace(mtcnn, "D:/bysj/sxt/image.jpg", trainObject.predict(['D:/bysj/sxt/image.jpg'], mtcnn=mtcnn, thresh=0.5, testLabel=testLabel, detectFace=DETECT_FACE))
    # pred = trainObject.predict(['D:/bysj/sxt/image.jpg'], mtcnn=mtcnn, thresh=0.5, testLabel=testLabel, detectFace=DETECT_FACE)
    # if pred == 1:
    #
    # # cap.release()                       # 关闭摄像头
    # # im = Image.open('D:/bysj/sxt/image.jpg')
    # # im = ImageTk.PhotoImage(im)
    # # label.image = im
    # # label.config(image=im)
    #     cropFace(mtcnn, "D:/bysj/sxt/image.jpg", 1)
    # else:
    #     cropFace(mtcnn, "D:/bysj/sxt/image.jpg", 0)


button1 = Button(root, text="拍照", font=("宋体", 25), fg="blue", command=get_photo)
button1.place(x=40, y=700)
button2 = Button(root, text="放上", font=("宋体", 25), fg="blue", command=imshow)
button2.place(x=300, y=700)
button3 = Button(root, text="删除", font=("宋体", 25), fg="blue", command=remove)
button3.place(x=450, y=700)


testLabel = np.array(testLabel, dtype="uint8")

# Calculate Test accuracy and produce classification report
# print(f'\nClassification Accuracy Obtained:: {accuracy_score(testLabel, pred)}\n')
#
# print("Classification Report::\n")
#
# print(classification_report(testLabel, pred))
root.mainloop()