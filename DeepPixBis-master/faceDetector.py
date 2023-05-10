import torch
from torchvision import transforms
from PIL import ImageDraw
import warnings
import numpy as np
from facenet_pytorch import MTCNN
from tkinter import *
import os
import cv2
from PIL import ImageTk
from PIL import Image
# root = Tk()
# root.geometry("600x450+374+182")
# label = Label(root, bg='red')
# # 定位
# label.place(x=0, y=0, width=600, height=350)


preprocess = transforms.Compose([
    transforms.ToPILImage(), #将张量转换为PIL图像
    transforms.Resize((224, 224)),
    transforms.ToTensor(), #oTensor()能够把灰度范围从0-255变换到0-1之间，而后面的transform.Normalize()则把0-1变换到(-1,1).
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #用均值和标准差归一化张量图像
    # 其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1
])

mtcnn = MTCNN()


def cropFace(mtcnn, PILImage, detectFace=True):

    r'''
    Utility to perform face extraction from raw images
    and return a tensor of shape `channels, 224, 224`
    从原始图像中执行人脸提取的实用程序
    并返回形状通道的张量，224224`
    '''
    name = PILImage
    PILImage = Image.open(PILImage)
    boxes, prob = mtcnn.detect(PILImage)
    # print(boxes, prob)
    a = ImageDraw.Draw(PILImage)
    i = 0

    # a.line(((204, 194), (524, 599)), fill=(255, 0, 0))
    if detectFace is True:
        img_cropped = mtcnn(PILImage, save_path=None)
    else:
        img_cropped = torch.tensor(np.array(PILImage))
    if img_cropped is None:
        print("[WARNING] Face not found for {}, skipping this image...".format(name))
        return None
    for i in range(1):
        if prob[i] < 0.8:
            continue
    a.line(((boxes[i][0], boxes[i][1]), (boxes[i][0], boxes[i][3])), fill=(0, 255, 0), width=5)
    a.line(((boxes[i][0], boxes[i][3]), (boxes[i][2], boxes[i][3])), fill=(0, 255, 0), width=5)
    a.line(((boxes[i][2], boxes[i][3]), (boxes[i][2], boxes[i][1])), fill=(0, 255, 0), width=5)
    a.line(((boxes[i][2], boxes[i][1]), (boxes[i][0], boxes[i][1])), fill=(0, 255, 0), width=5)
    # PILImage.show()
    # im = ImageTk.PhotoImage(PILImage)
    # label.image = im
    # label.config(image=im)
    # im = ImageTk.PhotoImage(PILImage)
    # label.image = im
    # label.config(image=im)
    img_cropped = preprocess(img_cropped)
    return img_cropped


def hh():
    cropFace(mtcnn, "D:/bysj/casia-fasd/test/1/3/3 10.jpg")
#
#
# button1 = Button(root, text="拍照", font=("宋体", 25), fg="blue", command=hh)
# button1.place(x=40, y=360)
# root.mainloop()


# if __name__ == "__main__":
#     mtcnn = MTCNN()
#     cropFace(mtcnn, "D:/bysj/sxt/image.jpg")
#     cropFace("./live/77fe7378-95ad-11ea-a79a-d7c6213d0492.jpg", None)