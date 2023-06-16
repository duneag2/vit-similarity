# -*- coding: utf-8 -*-
"""
Created on Fri July 29 2022
Last revised on Sat Feb 11 2023

@author: Anonymous

Cross-Sequence Learning: A Similarity based Learning Method for Multi-input Image Classification
1) Backbone network: ViT-L/32, ResNet50
2) Cross-Sequence Learning with cosine similarity

Refer to https://github.com/elinorwahl/pytorch-image-classifier
"""

import torch
from torch import nn
from torchvision import datasets, transforms, models
import torch.backends.cudnn as cudnn
import os.path
import argparse
import random
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import timm

from crseql import cross_sequence_learning

%matplotlib inline

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to input image folders')
    parser.add_argument('--save_dir', default='.', help='directory to save checkpoint')
    parser.add_argument('--arch', default='vit', type=str, choices=['vit', 'resnet50'], help='backbone network')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--fold', default=10, type=int, help='k-fold')
    parser.add_argument('--seed', default=256, type=int, help='fix the number of seed')
    parser.add_argument('--gpu', default=True, action='store_true', help='GPU usage')

    return parser.parse_args()


def dataset_loaders(data_dir, arch):

    if arch == 'vit':
        shape = 384
    elif arch == 'resnet50':
        shape = 1024
    
    # Note that image file name under 'train' and 'train_cropped' & 'test' and 'test_cropped' must be the same.
    train_dir = os.path.join(data_dir, 'train')
    train_cropped_dir = os.path.join(data_dir, 'train_cropped')
    test_dir = os.path.join(data_dir, 'test')
    test_cropped_dir = os.path.join(data_dir, 'test_cropped')

    train_transforms = transforms.Compose([transforms.Resize(shape),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(shape),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_cropped_datasets = datasets.ImageFolder(train_cropped_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_cropped_datasets = datasets.ImageFolder(test_cropped_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64)
    trainloader_cropped = torch.utils.data.DataLoader(train_cropped_datasets, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    testloader_cropped = torch.utils.data.DataLoader(test_cropped_datasets, batch_size=64)

    class_to_idx = train_datasets.class_to_idx

    return trainloader, trainloader_cropped, testloader, testloader_cropped, class_to_idx


def model_arch(arch):
    if arch == 'vit':
        model = timm.create_model('vit_base_patch32_384', pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError('{} is not supported.'.format(arch))

    return model


def build_classifier(arch, model, learning_rate):
    # transfer learning
    for parameter in model.parameters():
        parameter.requires_grad = False

    if arch == 'vit':
        num_features = model.head.in_features

    elif arch == 'resnet50':
        num_features = model.fc.in_features
    
    model.head = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : 0.95 ** epoch)

    return model, criterion, optimizer, scheduler

def multi_input(arch, model):
    model1 = model.copy()
    model2 = model.copy()

    if arch == 'vit':
        class TwoInputsNet(nn.Module):
            def __init__(self):
                super(TwoInputsNet, self).__init__()
                self.model1 = torch.nn.Sequential(*(list(model1.children())[:-4]), model1.blocks[:11], model1.blocks[-1].norm1).cuda()
                self.model2 = torch.nn.Sequential(*(list(model2.children())[:-4]), model2.blocks[:11], model2.blocks[-1].norm1).cuda()
                self.fc = nn.Sequential(
                    nn.Linear(294912, 1000),
                    nn.BatchNorm1d(1000),
                    nn.Dropout(0.3),
                    nn.Linear(1000, 100),
                    nn.BatchNorm1d(100),
                    nn.Dropout(0.3),
                    nn.Linear(100, 1),
                    nn.Sigmoid()
                )

            def forward(self, input1, input2):
                c = self.model1(input1)
                f = self.model2(input2)
                combined = torch.cat([c, f], dim = 2)
                combined2 = combined.reshape(c.shape[0], -1)
                out = self.fc(combined2)
                return out
        model_merged = TwoInputsNet().cuda()

    elif arch == 'resnet50':
        class TwoInputsNet(nn.Module):
            def __init__(self):
                super(TwoInputsNet, self).__init__()
                self.model1 = torch.nn.Sequential(*(list(model1.children())[:-2])).cuda()
                self.model2 = torch.nn.Sequential(*(list(model2.children())[:-2])).cuda()
                self.fc = nn.Sequential(
                    nn.Linear(1048576, 1000),
                    nn.BatchNorm1d(1000),
                    nn.Dropout(0.3),
                    nn.Linear(1000, 100),
                    nn.BatchNorm1d(100),
                    nn.Dropout(0.3),
                    nn.Linear(100, 1),
                    nn.Sigmoid()
                )

            def forward(self, input1, input2):
                c = self.model1(input1)
                f = self.model2(input2)
                combined = torch.cat([c, f], dim = 2)
                combined2 = combined.reshape(c.shape[0], -1)
                out = self.fc(combined2)
                return out
    
        model_merged = TwoInputsNet().cuda()
    
    return model_merged

def train(model, trainloader, trainloader_cropped, optimizer, log_interval, scheduler, criterion, device, epoch, batch):
    model.train()

    for _, (data1, data2) in enumerate(zip(trainloader, trainloader_cropped)):
        image1, label1 = data1
        image2, label2 = data2
        image1 = image1.to(device, dtype=torch.float)
        image2 = image2.to(device, dtype=torch.float)
        assert label1 == label2
        label1 = label1.to(device)
        optimizer.zero_grad()
        output = model(image1, image2).squeeze(dim=1)
        loss = criterion(output.to(torch.float32), label1.to(torch.float32))
        loss.backward()
        optimizer.step()

        if batch % log_interval == 0:
          print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(epoch, batch * len(image1), 
                len(trainloader.dataset), 100. * batch / len(trainloader), loss.item()))
    scheduler.step()

def evaluate(model, testloader, testloader_cropped, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    testlabel = []
    testpred = []
    testprob = []

    with torch.no_grad():
        for _, (data1, data2) in enumerate(zip(testloader, testloader_cropped)):
            image1, label1 = data1
            image2, label2 = data2
            image1 = image1.to(device, dtype=torch.float)
            image2 = image2.to(device, dtype=torch.float)
            assert label1 == label2
            label1 = label1.to(device)
            testlabel.append(label1)

            output = model(image1, image2).squeeze(dim=1)
            testprob.append(output.to(torch.float32))

            test_loss += criterion(output.to(torch.float32), label1.to(torch.float32)).item()
            output1 = output.cpu()
            output1[output1 >= 0.5] = 1
            output1[output1 < 0.5] = 0
            correct += output1.eq(label1.cpu()).int().sum()
            testpred.append(output1)
    
    test_loss /= len(testloader.dataset)
    test_accuracy = 100. * correct / len(testloader.dataset)
    return test_loss, test_accuracy, testlabel, testpred, testprob


def stat():



def main():
    in_args = get_input_args()
    device = torch.device('cuda' if torch.cuda.is_available() and in_args.gpu else 'cpu')

    torch.manual_seed(in_args.seed)
    torch.cuda.manual_seed(in_args.seed)
    torch.cuda.manual_seed_all(in_args.seed)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)

    data_dir = in_args.data_dir

    model = model_arch(in_args.arch)
    model, criterion, optimizer, scheduler = build_classifier(in_args.arch, model, in_args.learning_rate)
    model = multi_input(in_args.arch, model)
    trainloader, testloader, class_to_idx = dataset_loaders(data_dir)
    cross_sequence_learning(model, trainloader, validloader, criterion, optimizer, in_args.epochs, device)
    stat()

if __name__ == '__main__':
    main()