import torch
from torch.nn import BCELoss
from torch.optim import Adam
import numpy as np
import progressbar
# from sklearn.metrics import accuracy_score, classification_report
from statistics import mean
from faceDetector import cropFace

torch.cuda.set_device(0)
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda")
print(CUDA)
# CUDA = False
# DEVICE = torch.device("cpu")


class trainDeepPix(object):

    r'''
    Object to train DeepPix Network
    训练DeepPix网络的对象
    @params:: 
        `model`:: Initialized DeepPix Model
        `lr`:: Learning Rate for Adam Optimizer
        `weight_decay` :: L2 Regularization parameter
        `模型`：：已初始化DeepPix模型
        `lr`：：Adam Optimizer的学习率
        `weight_decay`：：L2正则化参数
    '''

    def __init__(self, model, lr, weight_decay):

        super(trainDeepPix, self).__init__()

        self.model = model
        self.lossC = BCELoss()
        self.lossS = BCELoss()

        if CUDA:
            self.model = self.model.cuda()
            self.lossC = self.lossC.cuda()
            self.lossS = self.lossS.cuda()

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _convertYLabel(y):

        returnY = torch.ones((y.shape[0], 196)).type(torch.FloatTensor)
        returnY = returnY.cuda()

        for i in range(y.shape[0]):
            returnY[i] = returnY[i]*y[i]

        return returnY.cuda() if CUDA else returnY

    @staticmethod
    def _calcAccuracy(yPred, yTrue):

        yPred = yPred.view(-1)
        yTrue = yTrue.view(-1)

        if CUDA:
            yPred = (yPred > 0.5).type(torch.cuda.LongTensor)
        else:
            yPred = (yPred > 0.5).type(torch.LongTensor)

        return torch.sum((yTrue == yPred).type(torch.LongTensor), dim=0)/float(yTrue.shape[0])

    def train(self, ImgList, LabelList, mtcnn, batch_size=32, epochs=50, detectFace=True):

        r'''
        Utility to train DeepPix Model,
        @params:: 
            `ImgList`:: List of Image Paths
            `LabelList`:: List of Labels correspoding to images
                            [should be 0 or 1]
        '''
        
        ImgArray = []

        widgets = [f"Cropping faces: " if detectFace else f"Preprocessing Data: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]

        pbar = progressbar.ProgressBar(maxval=len(ImgList), widgets=widgets).start()

        __ctr = 0
        
        for img in ImgList:
            res = cropFace(mtcnn, img, detectFace=detectFace)
            if res is not None:
                ImgArray.append(res.unsqueeze(0))
            else:
                del LabelList[__ctr]

            pbar.update(__ctr)
            __ctr+=1

        pbar.finish()

        for epoch in range(epochs):

            widgets = [f"Epoch {epoch+1}/{epochs} ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]

            pbar = progressbar.ProgressBar(maxval=np.arange(0, len(ImgArray), batch_size).shape[0], widgets=widgets).start()

            __ctr = 0

            batch_loss = []
            batch_accClass = []
            batch_accSeg = []

            for item in np.arange(0, len(ImgArray), batch_size):

                trainX = torch.cat(ImgArray[item:item+batch_size], dim=0)
                trainY = torch.tensor(LabelList[item:item+batch_size]).type(torch.FloatTensor)

                if CUDA:
                    trainX = trainX.cuda()
                    trainY = trainY.cuda()

                self.model.train()
                self.optimizer.zero_grad()

                classPred, segPred = self.model(trainX)

                segPred = segPred.view(trainX.shape[0], -1)

                train_loss = self.lossC(classPred.squeeze(), trainY) + self.lossC(segPred.squeeze(), self._convertYLabel(trainY))

                train_loss.backward()
                self.optimizer.step()

                classAcc = self._calcAccuracy(classPred, trainY)
                SegAcc = self._calcAccuracy(segPred, self._convertYLabel(trainY))

                batch_loss.append(train_loss.item())
                batch_accClass.append(classAcc.item())
                batch_accSeg.append(SegAcc.item())
                pbar.update(__ctr)
                __ctr+=1

            pbar.finish()

            print(f'Summary -> train_loss:: {mean(batch_loss)}, class_acc:: {mean(batch_accClass)}, seg_acc:: {mean(batch_accSeg)}')

    def predict(self, ImgList, mtcnn, batch_size=16, thresh=0.5, testLabel=None, detectFace=True):

        r'''
        Utility to predict `spoof/bonafide` viz `0/1` given list
        of test image Path

        @params:: 
            `ImgList`:: Test Image Path List
            `mtcnn`:: Face Cropping Module
            `batch size`:: Batch Size for testing
            `thresh`:: Threshold to classify an image as spoof or bonafide
        '''

        self.model.eval()

        ImgArray = []

        widgets = [f"Cropping faces: " if detectFace else f"Preprocessing Data: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]

        pbar = progressbar.ProgressBar(maxval=len(ImgList), widgets=widgets).start()

        __ctr = 0
        
        for img in ImgList:

            res = cropFace(mtcnn, img, detectFace=detectFace)
            if res is not None:
                ImgArray.append(res.unsqueeze(0))
            else:
                if testLabel is not None:
                    del testLabel[__ctr]

            pbar.update(__ctr)
            __ctr+=1

        # ImgArray = torch.cat(ImgArray, dim=0)

        pbar.finish()

        returnY = np.zeros((len(ImgArray)), dtype="uint8")

        widgets = [f"Predicting ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]

        pbar = progressbar.ProgressBar(maxval=np.arange(0, len(ImgArray), batch_size).shape[0], widgets=widgets).start()

        __ctr = 0

        for item in np.arange(0, len(ImgArray), batch_size):

            _, segPred = self.model(torch.cat(ImgArray[item:item+batch_size], dim=0).cuda() if CUDA else torch.cat(ImgArray[item:item+batch_size], dim=0))

            segPred = segPred.view(segPred.shape[0], -1)

            segPred = torch.mean(segPred, dim=1)
            segPred = (segPred > thresh).type(torch.LongTensor).cpu().detach().numpy()

            returnY[item:item+batch_size] = segPred
            pbar.update(__ctr)
            __ctr += 1

        pbar.finish()

        return returnY

    def saveModel(self, path):
        r'''
        Saves current model state to the path given
        '''

        torch.save(self.model.state_dict(), path)
        print("[INFO] Model Saved")

    def loadModel(self, path):
        r'''
        Loads model state from the path given
        and maps to available/given device
        '''

        self.model.load_state_dict(torch.load(path, map_location=DEVICE if CUDA else torch.device("cpu")))
        print("[INFO] Model Loaded..")

