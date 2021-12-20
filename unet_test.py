import numpy as np
import torch
import cv2
import argparse
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from torchvision.utils import save_image
from stackimage import stackImages
from unet import Unet
from skimage import img_as_ubyte
from dataset import LiverDataset



def train_model(model, criterion, optimizer, dataload, num_epochs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            #print('++++++++++++++++++++++++++++')
            #print(x.shape)
            #print(y.shape)
            #print('---------------------------')
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #loss = nn.BCEWithLogitsloss(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model


# 训练模型
def train(epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  # ->[-1,1]
    ])
    y_transforms = transforms.ToTensor()
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, epochs)


            
# 显示模型的输出结果
def seg(img):
    args = make_prepare()
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  # ->[-1,1]
    ])
    model.eval()
    imgs = []
    if('.jpg' not in img):
        if(os.listdir(img)!=''):
            for i in os.listdir(img):
                path = os.path.join(img, i)
                imgs.append(path)
        else:
            return
    else:
        imgs.append(img)
    i = 0
    
    with torch.no_grad():
        for x in imgs:
            if('.jpg' not in x):
                continue
            print('load:', x)
            x = x_transforms(Image.open(x).convert('RGB').resize((512, 512)))
            x = torch.unsqueeze(x, 0)
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = img_as_ubyte(img_y)
            img_y[img_y>30] = 255
            img_y[img_y<=30] = 0
            imgStack = img_y
            cv2.imwrite(f'seg_result_img/{i}.jpg',imgStack)
            i = i + 1


# 参数
def make_prepare():
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--img", type=str, default='data/sample/000.jpg')
    parse.add_argument("--train", type=int, default=0)
    parse.add_argument("--ckp", type=str, default='weights_199.pth')
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    
    args = make_prepare()
    
    nums_epochs = 200  
    
    # train
    if args.train:
        train(nums_epochs)
    #test
    else:
        seg(args.img)
