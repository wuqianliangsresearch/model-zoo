# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:44:04 2019

@author: wuqianliang
"""


import torch
import torch.backends.cudnn as cudnn
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
    
root="./"
 
# -----------------ready the dataset--------------------------
def opencvLoad(imgPath,resizeH,resizeW):
    image = cv2.imread(imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))  
    image = torch.from_numpy(image)
    return image
    
class LoadPartDataset(Dataset):
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
        img = opencvLoad(image,224,224)
        return img,label
    def __len__(self):
        return len(self.imgs)
        
trainSet =LoadPartDataset(txt=root+'train.txt')
test_data=LoadPartDataset(txt=root+'val.txt')

train_loader = DataLoader(dataset=trainSet, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
 
 
#-----------------create the Net and training------------------------
 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 0),
            torch.nn.ReLU(),
            # k = 2, n = 5, alpha= 0.0001, and  beta= 0.75
            torch.nn.LocalResponseNorm(size = 5, alpha=1e-4, beta=0.75, k=2.),
            torch.nn.MaxPool2d(3,2)
        )
        # group  conv
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2, groups=2),
            torch.nn.ReLU(),
            # k = 2, n = 5, alpha= 0.0001, and  beta= 0.75
            torch.nn.LocalResponseNorm(size = 5, alpha=1e-4, beta=0.75, k=2.),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        # gourp conv
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384,384, 3, 1, 1, groups=2),
            torch.nn.ReLU(),
        )
        # group conv
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384,256, 3, 1, 1, groups=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 1000)
        )
 
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        # flatten
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out
 
 
model = Net()
 
finetune = None
finetune = r'./model/_iter_99.pth'
 
if finetune is not None:
    print( '[0] Load Model {}'.format(finetune))
 
    pretrained_dict = model.state_dict()
    finetune_dict = torch.load(finetune)
 
    # model_dict = torch.load(finetune)
    # pretrained_dict = net.state_dict()
 
    model_dict = {k: v for k, v in finetune_dict.items() if k in pretrained_dict}
    pretrained_dict.update(model_dict)
 
    model.load_state_dict(pretrained_dict)
 
#model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.cuda()
cudnn.benchmark = True
print(model)
 
 
#optimizer = torch.optim.Adam(model.parameters())
#loss_func = torch.nn.CrossEntropyLoss()
 
# updata net
lr = 1e-5
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
 
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
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        trainSet)), train_acc / (len(trainSet))))
 
    if (epoch + 1) % 10 == 0:
        sodir = './model/_iter_{}.pth'.format(epoch)
        print '[5] Model save {}'.format(sodir)
        torch.save(model.module.state_dict(), sodir)
 
    # adjust
    if (epoch + 1)% 100 == 0:
        lr = lr / 10
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
 
    # evaluation--------------------------------
    # model.eval()
    # eval_loss = 0.
    # eval_acc = 0.
    # for trainData, trainLabel in test_loader:
    #     trainData, trainLabel = Variable(trainData, volatile=True), Variable(trainLabel, volatile=True)
    #     out = model(trainData)
    #     loss = loss_func(out, trainData)
    #     eval_loss += loss.data[0]
    #     pred = torch.max(out, 1)[1]
    #     num_correct = (pred == trainData).sum()
    #     eval_acc += num_correct.data[0]
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
