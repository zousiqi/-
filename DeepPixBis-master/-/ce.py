from tkinter import *
import os

import numpy as np
import torch
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
root.geometry("600x450+374+182")

root.title("我的个性签名设计")

# 添加标签控件

label = Label(root, fg='red')
# 定位
label.place(x=0, y=0, width=600, height=350)

# cap = cv2.VideoCapture(0)  # 开启摄像头

cap = cv2.VideoCapture("http://admin:admin@192.168.43.1:8081/")


# def ce():
#     cropFace(mtcnn, "D:/bysj/casia-fasd/test/1/3/3 10.jpg")


def remove():
    global ji
    ji = 1
    # label.config(text="aaaaaaaa", fg='green')
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
            img = ImageTk.PhotoImage(img)
            label.image = img
            # label['image'] = img
            label.config(image=img)
    #创建一个定时器，每10ms进入一次函数

    root.after(10, imshow)


def cropFace(mtcnn, PILImage):

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
    draw.text((100, 100), "邹思琪", font=setFont, fill=(0, 255, 0))  # 利用ImageDraw的内置函数，在图片上写入文字
    PILImage.save("D:/bysj/sxt/image.jpg")
    # img_open = Image.open('图片')
    # resized = img_open.resize((600, 700), Image.ANTIALIAS)  # 设置图片尺寸
    # img = ImageTk.PhotoImage(resized)
    # self.label.config(image=img)
    # self.label.image = img
    im = Image.open("D:/bysj/sxt/image.jpg")
    im = im.resize((600, 350), Image.ANTIALIAS)
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
    # cap.release()                       # 关闭摄像头
    # im = Image.open('D:/bysj/sxt/image.jpg')
    # im = ImageTk.PhotoImage(im)
    # label.image = im
    # label.config(image=im)
    cropFace(mtcnn, "D:/bysj/sxt/image.jpg")


button1 = Button(root, text="拍照", font=("宋体", 25), fg="blue", command=get_photo)
button1.place(x=40, y=360)
button2 = Button(root, text="放上", font=("宋体", 25), fg="blue", command=imshow)
button2.place(x=300, y=360)
button3 = Button(root, text="删除", font=("宋体", 25), fg="blue", command=remove)
button3.place(x=450, y=360)

root.mainloop()
