# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:44:04 2019

@author: wuqianliang
"""

from __future__ import unicode_literals, print_function, division
from io import open

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn
#import cv2 
from torch.utils.data import Dataset, DataLoader


root="./caffe_ilsvrc12/"

#Image transforming helper functions

# horizontal reflections
def flip(self, flag="h"):
    generate_img = np.zeros(self.img.shape)
    if flag == "h":
        for i in range(self.h):
            for j in range(self.w):
                generate_img[i, self.h - 1 - j] = self.img[i, j]
    else:
        for i in range(self.h):
            for j in range(self.w):
                generate_img[self.h-1-i, j] = self.img[i, j]
    return generate_img
        
# Rotate the image by angle
def image_rotate(img, angle):
    rows, cols, dims = img.shape
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, matrix, (cols, rows))

# Adjust the image size by scale
def image_scale(img, scale):
    rows, cols, dims = img.shape
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
    return cv2.warpAffine(image, matrix, (cols, rows))

# Translate image by the value of x and y
def image_translate(img, x, y):
    rows, cols, dims = img.shape
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, matrix, (cols, rows))

# Shear image randomly by the factor of shear_range
def image_shear(img, shear_range):
    rows, cols, dims = img.shape
    
    pts1 = np.float32([[5, 5],[20, 5],[5, 20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, matrix, (cols, rows))

# Adjust the brightness of images by factor
def image_brightness(image, factor):
    
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.minimum(hsv[:,:,2] * factor, 200)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

# Center normalize
def center_normalize(img):
    
    img = img.astype('float32')
    img = (img - 128)/128
    return img

def PCA_AUG(img):
    original_image=cv.imread(img)
    #first you need to unroll the image into a nx3 where 3 is the no. of colour channels
    renorm_image = np.reshape(original_image,(original_image.shape[0]*original_image.shape[1],3))
    #Before applying PCA you must normalize the data in each column separately as we will be applying PCA column-wise
    mean = np.mean(renorm_image, axis=0) #computing the mean
    std = np.std(renorm_image, axis=0) #computing the standard deviation
    renorm_image = renorm_image.astype('float32') #we change the datatpe so as to avoid any warnings or errors
    renorm_image -= np.mean(renorm_image, axis=0)
    renorm_image /= np.std(renorm_image, axis=0) # next we normalize the data using the 2 columns
    cov = np.cov(renorm_image, rowvar=False) #finding the co-variance matrix for computing the eigen values and eigen vectors.
    lambdas, p = np.linalg.eig(cov) # finding the eigen values lambdas and the vectors p of the covarince matrix
    alphas = np.random.normal(0, 0.1, 3) # aplha here is the gaussian random no. generated
    delta = np.dot(p, alphas*lambdas) #delta here represents the value which will be added to the re_norm image
    pca_augmentation_version_renorm_image = renorm_image + delta #forming augmented normalised image
    pca_color_image = pca_augmentation_version_renorm_image * std + mean #de-normalising the image
    pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype('uint8') # necessary conditions which need to be checked
    pca_color_image=np.ravel(pca_color_image).reshape((original_image.shape[0],original_image.shape[1],3)) #rollong back the image into a displayable just as original image    
    return pca_color_image
    
# -----------------ready the dataset--------------------------
def opencvLoad(imgPath,resizeH,resizeW):
    image = cv2.imread(imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))  
    image = torch.from_numpy(image)
    return image
    
class LoadPartDataset(Dataset,flag='train'):
    def __init__(self, txt):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            labelList = int(words[1])
            imageList = words[0]
            imgs.append((imageList, labelList))
        self.imgs = imgs
            
    def __getitem__(self, item):
        image, label = self.imgs[item]
           
        image = '/home/wuqianliang/ILSVRC2012_img_train'.join(image)
        
        if flag is 'val':
            image = '/home/wuqianliang/ILSVRC2012_img_val'.join(image)
        elif flag is 'test':
            image = '/home/wuqianliang/ILSVRC2012_img_test'.join(image)
        img = opencvLoad(image,227,227)
        return img,label
    def __len__(self):
        return len(self.imgs)
        
trainSet =LoadPartDataset(txt=root+'train.txt')
test_data=LoadPartDataset(txt=root+'val.txt', flag='val')
#
train_loader = DataLoader(dataset=trainSet, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
 
 
#-----------------create the Net and training------------------------
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0),
            nn.ReLU(),
            # k = 2, n = 5, alpha= 0.0001, and  beta= 0.75
            nn.LocalResponseNorm(size = 5, alpha=1e-4, beta=0.75, k=2.),
            nn.MaxPool2d(3,2)
        )
        # group  conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2, groups=2),
            nn.ReLU(),
            # k = 2, n = 5, alpha= 0.0001, and  beta= 0.75
            nn.LocalResponseNorm(size = 5, alpha=1e-4, beta=0.75, k=2.),
            nn.MaxPool2d(3,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256,384, 3, 1, 1),
            nn.ReLU(),
        )
        # gourp conv
        self.conv4 = nn.Sequential(
            nn.Conv2d(384,384, 3, 1, 1, groups=2),
            nn.ReLU(),
        )
        # group conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(384,256, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        self.dense = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )
        
#        init.xavier_normal(self.conv1.parameters)
#        init.normal(self.conv2.parameters,mean=0., std=0.5)
#        init.normal(self.conv3.parameters,mean=0., std=0.5)
#        init.normal(self.conv4.parameters,mean=0., std=0.5)
#        init.normal(self.conv5.parameters,mean=0., std=0.5)
#        init.normal(self.dense.parameters,mean=0., std=0.5)
 
    def forward(self, x):

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        # flatten by batch
        res = conv5_out.view(conv5_out.size(0), -1).float()
        out = self.dense(res)
        return out
 
 
model = Net()

# 参数初始化
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        print(layer)
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)).float()
        
finetune = None
#finetune = r'./model/_iter_99.pth'
 
if finetune is not None:
    print( '[0] Load Model {}'.format(finetune))
 
    pretrained_dict = model.state_dict()
    finetune_dict = torch.load(finetune)
 
    model_dict = {k: v for k, v in finetune_dict.items() if k in pretrained_dict}
    pretrained_dict.update(model_dict)
 
    model.load_state_dict(pretrained_dict)
 
#model = torch.nn.DataParallel(model, device_ids=[0])
model.cuda()
cudnn.benchmark = True
#print(model)
 
 
#optimizer = torch.optim.Adam(model.parameters())
#loss_func = torch.nn.CrossEntropyLoss()
 

if __name__ == '__main__':
    
#    input_tensor =  torch.randn(( 64, 3, 227, 227)).cuda()
#
#    output = model(input_tensor)
#    print(output)
#    exit()
    
    # updata net
    lr = 1e-5
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    for epoch in range(10000):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for trainData, trainLabel in train_loader:
            trainData, trainLabel = trainData.cuda(), trainLabel.cuda()
            out = model(trainData)
            loss = loss_func(out, trainLabel)
            train_loss += loss.data[0]
            pred = torch.max(out, 1)[1]
            train_correct = (pred == trainLabel).sum()
            train_acc += train_correct.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #  if epoch % 100 == 0:
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(trainSet)), train_acc / (len(trainSet))))
     
        if (epoch + 1) % 10 == 0:
            sodir = './model/_iter_{}.pth'.format(epoch)
            print( '[5] Model save {}'.format(sodir))
            torch.save(model.module.state_dict(), sodir)
     
        # adjust
        if (epoch + 1)% 100 == 0:
            lr = lr / 10
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
