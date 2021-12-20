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

# 是否使用cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把多个步骤整合到一起, channel=（channel-mean）/std, 因为是分别对三个通道处理
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  # ->[-1,1]
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

# 参数解析器,用来解析从终端读取的命令
parse = argparse.ArgumentParser()


def train_model(model, criterion, optimizer, dataload, num_epochs=200):
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
def train():
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# 显示模型的输出结果
def test_1():
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    #import matplotlib.pyplot as plt
    #plt.ion()
    imgs = []
    root = "data/val"
    n = len(os.listdir(root)) // 2
    
    print('n:',n)
    for i in range(n):
        img = os.path.join(root, "%03d.jpg" % i)
        print(img)
        # mask = os.path.join(root, "%03d_mask.jpg" % i)
        imgs.append(img)
    i = 0
    with torch.no_grad():
        for x, _ in dataloaders:
            print('++++++++++++++++++++++++++')
            print(x)
            print('++++++++++++++++++++++++++')
            y = model(x)
            img_x = torch.squeeze(_).numpy()
            img_y = torch.squeeze(y).numpy()
            img_input = cv2.imread(imgs[i],cv2.IMREAD_COLOR)
            #im_color = cv2.applyColorMap(img_input, cv2.COLORMAP_JET)
            img_x = img_as_ubyte(img_x)
            img_y = img_as_ubyte(img_y)
            #imgStack = stackImages(0.8, [[img_input, img_x, img_y]])
            # 转为伪彩色，视情况可以加上
            #imgStack = cv2.applyColorMap(imgStack, cv2.COLORMAP_JET)
            imgStack = img_y
            cv2.imwrite(f'train_img/{i}.jpg',imgStack)
            #plt.imshow(imgStack)
            i = i + 1
            #plt.pause(0.1)
            
def seg(img):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    # liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    # dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    #import matplotlib.pyplot as plt
    #plt.ion()
    imgs = []
    
    imgs.append(img)
    i = 0
    
    with torch.no_grad():
        for x in imgs:
            x = x_transforms(Image.open(x).convert('RGB').resize((512, 512)))
            x = torch.unsqueeze(x, 0)
            y = model(x)
            # img_x = torch.squeeze(_).numpy()
            img_y = torch.squeeze(y).numpy()
            # img_input = cv2.imread(imgs[i],cv2.IMREAD_COLOR)
            #im_color = cv2.applyColorMap(img_input, cv2.COLORMAP_JET)
            # img_x = img_as_ubyte(img_x)
            img_y = img_as_ubyte(img_y)
            #imgStack = stackImages(0.8, [[img_input, img_x, img_y]])
            # 转为伪彩色，视情况可以加上
            #imgStack = cv2.applyColorMap(imgStack, cv2.COLORMAP_JET)
            img_y[img_y>30] = 255
            img_y[img_y<=30] = 0
            imgStack = img_y
            cv2.imwrite(f'train_img/{i}.jpg',imgStack)
            #plt.imshow(imgStack)
            i = i + 1
            #plt.pause(0.1)


# 参数
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--batch_size", type=int, default=1)
parse.add_argument("--ckp", type=str, help="the path of model weight file")
args = parse.parse_args()

# train
#train()

#test()
args.ckp = "weights_199.pth"
# test_1()
