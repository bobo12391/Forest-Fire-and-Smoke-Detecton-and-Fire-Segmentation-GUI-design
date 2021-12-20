#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 19:59:59 2021

@author: yhf
"""


import sys
import PyQt5
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QPushButton, QWidget, QApplication, QMessageBox
import os
import cv2
from detect import detect
from unet_test import seg


#基本窗口使用函数类
class Basic_func(QWidget):
    def __init__(self):
        super().__init__()
        
    #按钮初始化
    def btn_self(self, btn, size, fontt='Times New Roman', btn_style_text=None):  
        font = QtGui.QFont()
        font.setFamily(fontt)
        font.setPointSize(size) 
        btn.setFont(font)
        if btn_style_text:
            btn.setStyleSheet(btn_style_text)
            
    #字体初始化    
    def font_sel(self, label, fontt, size):
        font = QtGui.QFont()
        font.setFamily(fontt)
        font.setPointSize(size) 
        label.setFont(font)
        
    #不带框label初始化
    def label_self(self, label, x, y, size, fontt="Times New Roman", label_style_text=None):
        label.move(x, y)
        self.font_sel(label, fontt, size)
        if label_style_text:
            label.setStyleSheet(label_style_text)
        return label
            
    #带框label初始化
    def label_self2(self, label, x1, y1, x2, y2, label_style_text=None):
        label.setFixedSize(x1, y1)
        label.move(x2, y2)
        if label_style_text:
            label.setStyleSheet(label_style_text)
        return label

#页面使用按键初始化
class Button(QWidget):
    def __init__(self):
        super().__init__()
        self.btn = QPushButton(self)
        self.btn2 = QPushButton(self)
        self.btn_seg_test = QPushButton(self)
        self.btn_stop1 = QPushButton(self)
        self.btn_stop2 = QPushButton(self)
        self.btn_camera = QPushButton(self)
        self.btn_leave = QPushButton(self)
        self.btn_test = QPushButton(self)
        self.btn_reset = QPushButton(self)
        self.btn_up = QPushButton(self)
        self.btn_down = QPushButton(self)
        self.btn_left = QPushButton(self)
        self.btn_right = QPushButton(self)
        self.btn_medium = QPushButton(self)
        self.set_text_1()
        
    def set_text_1(self):
        self.btn.setText("Insert a Video or Picture")
        self.btn2.setText("Start to Detect")
        self.btn_seg_test.setText("Start to Segment")
        self.btn_stop1.setText("Stop")
        self.btn_stop2.setText("Stop")
        self.btn_camera.setText("Open the Camera")
        self.btn_leave.setText("Quit")
        self.btn_test.setText("Test")
        self.btn_reset.setText("Reset")
        self.btn_up.setText(" ↑ ")
        self.btn_down.setText(" ↓ ")
        self.btn_left.setText("←")
        self.btn_right.setText("→")
        self.btn_medium.setText(" O ")
        
#页面使用label初始化
class Label(QWidget):
    
    def __init__(self):
        super().__init__()
        self.title_label = PyQt5.QtWidgets.QLabel(self)
        self.version_label = PyQt5.QtWidgets.QLabel(self)
        self.copyright_label = PyQt5.QtWidgets.QLabel(self)
        self.load1_label = PyQt5.QtWidgets.QLabel(self)
        self.load2_label = PyQt5.QtWidgets.QLabel(self)
        self.detect_label = PyQt5.QtWidgets.QLabel(self)
        self.address_label = PyQt5.QtWidgets.QLabel(self)
        self.cloud_label = PyQt5.QtWidgets.QLabel(self)
        self.Author_label = PyQt5.QtWidgets.QLabel(self)
        self.R_rate_label = PyQt5.QtWidgets.QLabel(self)
        self.S_rate_label = PyQt5.QtWidgets.QLabel(self)
        self.label = PyQt5.QtWidgets.QLabel(self)
        self.label2 = PyQt5.QtWidgets.QLabel(self)
        self.label3 = PyQt5.QtWidgets.QLabel(self)
        self.set_text_2()
        
    def set_text_2(self):
        self.title_label.setText('Forest Fire or Smoke Segmentation and Detection')
        self.version_label.setText('Version:0.0.2')
        self.copyright_label.setText('Copyright © East China University of Science and Technology')
        self.load1_label.setText('Institute of Information Science and Engineering')
        self.load2_label.setText('Department of Information and Communication Engineering')
        self.detect_label.setText('Enter the URL for Online Detection :')
        self.address_label.setText('Enter the File Address for Local Test:')
        self.cloud_label.setText('Cloud Operation')
        self.Author_label.setText('Author:Haofei Yuan')
        # self.R_rate_label.setText('Recognition Rate:')
        # self.S_rate_label.setText('Segmentation Accuracy:')
        self.label.setText("                           Insert a Picture/Video  ")
        self.label2.setText("                           Detection Results  ")
        self.label3.setText("                        Segmentation Results  ")


#计时器类(2个定时器)
class Timer(Button, Label):
    
    def __init__(self):
        super().__init__()
        self.cap1 = []
        self.frame1 = [] 
        self.timer_camera1 = QTimer()     #定义定时器
        self.cap2 = []
        self.frame2 = [] 
        self.timer_camera2 = QTimer()     #定义定时器   
        self.video_flag = 0
        self.openfile_name = []
        
    def openFrame1(self):
        if(self.cap1.isOpened()): 
            ret, self.frame1 = self.cap1.read()
            if ret:
                frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame1.shape
                bytesPerLine = bytesPerComponent * width
                q_image1 = QtGui.QImage(frame1.data,  width, height, bytesPerLine,
                                 QtGui.QImage.Format_RGB888).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QtGui.QPixmap.fromImage(q_image1))
            else:
                self.cap1.release()
                self.timer_camera1.stop() 
                
    def openFrame2(self):
        self.btn_stop2.setHidden(False)
        if(self.cap2.isOpened()): 
            ret, self.frame2 = self.cap2.read()
            if ret:
                frame2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame2.shape
                bytesPerLine = bytesPerComponent * width
                q_image2 = QtGui.QImage(frame2.data,  width, height, bytesPerLine,
                                 QtGui.QImage.Format_RGB888).scaled(self.label2.width(), self.label2.height())
                self.label2.setPixmap(QtGui.QPixmap.fromImage(q_image2))
            else:
                self.cap2.release()
                self.timer_camera2.stop() 
        
    def slotStart1(self):
        videoName, _ = self.openfile_name
        if videoName != "":
            self.cap1 = cv2.VideoCapture(videoName)
            self.timer_camera1.start(100)
            self.timer_camera1.timeout.connect(self.openFrame1)

         
    def slotStart2(self, name):
        self.video_flag = 0
        videoName = name
        if videoName != "":
            self.cap2 = cv2.VideoCapture(videoName)
            self.timer_camera2.start(100)
            self.timer_camera2.timeout.connect(self.openFrame2)
        
    def slotStop1(self):
        if self.cap1 != []:
            self.cap1.release()
            self.timer_camera1.stop()   # 停止计时器
            self.label.setText("                        This video has been stopped")
            self.btn_stop1.setHidden(True)
        else:
            self.label_num.setText("Push the left upper corner button to Quit.")

    def slotStop2(self):
        if self.cap2 != []:
            self.cap2.release()
            self.timer_camera2.stop()   # 停止计时器
            self.label2.setText("                   This video has been stopped")
            self.btn_stop2.setHidden(True)
        else:
            self.label_num.setText("Push the left upper corner button to Quit.")
        

#主窗体
class Window(Timer, Basic_func):
    def __init__(self, detect_permit=1, seg_permit=1, camera_permit=1):
        super().__init__()
        self.file_dir = ''
        self.flag = 0
        self.flag_test = 0
        self.flag_video_test = 0
        self.flag_seg = 0
        self.win()
        self.btn_init()
        self.label_init()
        self.label_init()
        self.detect_permit = detect_permit
        self.seg_permit = seg_permit
        self.camera_permit = camera_permit
        
    
    #总窗口大小设置
    def win(self):
        self.setWindowTitle(" ")
        self.resize(1650, 1020)
        self.setFixedSize(1650, 1020)
        self.setStyleSheet("background-color: linen")
    
    #按键初始化
    def btn_init(self):   
        # Insert a Video or Picture
        self.btn_self(self.btn, 20)
        self.btn.move(320, 640)
        self.btn.clicked.connect(self.file_select)
        # Start to Detect
        self.btn_self(self.btn2, 20, btn_style_text='background-color:lightgreen')
        self.btn2.move(360, 700)
        self.btn2.clicked.connect(self.detect_test)
        # Start to Segment
        self.btn_self(self.btn_seg_test, 20, btn_style_text='background-color:lightgreen')
        self.btn_seg_test.move(20, 700)
        self.btn_seg_test.clicked.connect(self.seg_test)
        # Stop(左边视频播放窗口)
        self.btn_stop1.move(50, 600)
        self.btn_stop1.clicked.connect(self.slotStop1)
        self.btn_stop1.setHidden(True)
        # Stop(右边检测视频播放窗口)
        self.btn_stop2.move(930, 450)
        self.btn_stop2.clicked.connect(self.slotStop2)
        self.btn_stop2.setHidden(True)
        # Open the Camera
        self.btn_self(self.btn_camera, 20, btn_style_text='background-color:violet')
        self.btn_camera.move(675, 700)
        self.btn_camera.clicked.connect(self.shell)
        # Quit
        self.btn_self(self.btn_leave, 20, btn_style_text='background-color:dodgerblue')
        self.btn_leave.move(400, 770)
        self.btn_leave.clicked.connect(self.close)
        # Test
        self.btn_self(self.btn_test, 20, btn_style_text='background-color:lightgreen')
        self.btn_test.move(20, 865)
        # Reset
        self.btn_self(self.btn_reset, 20, btn_style_text='background-color:orange')
        self.btn_reset.move(20, 770)
        self.btn_reset.clicked.connect(self.reset)
        # ↑
        self.btn_self(self.btn_up, 12, btn_style_text='color:white;background-color: black;border-radius: 10px;border: 8px groove gray;border-style: outset')
        self.btn_up.move(755, 837)
        # ↓
        self.btn_self(self.btn_down, 12, btn_style_text='color:white;background-color: black;border-radius: 10px;border: 8px groove gray;border-style: outset')
        self.btn_down.move(755, 903)
        # ←
        self.btn_self(self.btn_left, 12, btn_style_text='color:white;background-color: black;border-radius: 10px;border: 8px groove gray;border-style: outset')
        self.btn_left.move(720, 870)  
        # →
        self.btn_self(self.btn_right, 12, btn_style_text='color:white;background-color: black;border-radius: 10px;border: 8px groove gray;border-style: outset')
        self.btn_right.move(790, 870)
        # O
        self.btn_self(self.btn_medium, 15, btn_style_text='background-color: rgb(192, 192, 192);border-radius: 10px; border: 4px groove gray;border-style: outset')
        self.btn_medium.move(755, 872)
        
    
    #label初始化
    def label_init(self):
        self.title_label = self.label_self(self.title_label, 350, 0, 36)
        self.version_label = self.label_self(self.version_label, 1500, 975, 18, label_style_text="color:black")
        self.copyright_label = self.label_self(self.copyright_label, 20, 940, 18, label_style_text="color:black")
        self.load1_label = self.label_self(self.load1_label, 150, 975, 18, label_style_text="color:black")
        self.load2_label = self.label_self(self.load2_label, 730, 940, 18, label_style_text="color:black")
        self.detect_label = self.label_self(self.detect_label, 20, 840, 18, label_style_text="color:red")
        self.address_label = self.label_self(self.address_label, 20, 900, 18, label_style_text="color:green")
        self.cloud_label = self.label_self(self.cloud_label, 700, 790, 15, label_style_text='border-width: 3px;border-style: solid;border-color: black;color:black')
        self.Author_label = self.label_self(self.Author_label, 730, 975, 18, label_style_text="color:black")
        self.R_rate_label = self.label_self(self.R_rate_label, 900, 500, 20, label_style_text="color:red")
        self.S_rate_label = self.label_self(self.S_rate_label, 1300, 500, 20, label_style_text="color:red")
        self.label = self.label_self2(self.label, 850, 550, 20, 75, label_style_text="QLabel{background:Black;color:rgb(255, 250, 250, 120);font-size:35px;font-weight:bold;font-family:Times New Roman;}")
        self.label2 = self.label_self2(self.label2, 725, 400, 900, 75, label_style_text="QLabel{background:Black;color:rgb(255, 250, 250, 120);font-size:35px;font-weight:bold;font-family:Times New Roman;}")
        self.label3 = self.label_self2(self.label3, 725, 375, 900, 550, label_style_text="QLabel{background:Black;color:rgb(255, 250, 250, 120);font-size:35px;font-weight:bold;font-family:Times New Roman;}")
    
    #输入框初始化
    def line_init(self):
        
        def setup_line(text, x1, y1, x2, y2, font_size):
            le = PyQt5.QtWidgets.QLineEdit(self) 
            le.setPlaceholderText(text)
            le.setEchoMode(PyQt5.QtWidgets.QLineEdit.Normal)
            le.move(x1, y1)
            le.setFixedSize(x2, y2)
            font = QtGui.QFont()
            font.setPointSize(font_size) 
            le.setFont(font)     
            return le
        
        self.line1 = self.setup_line("https://xxx", 380, 840, 310, 30, 15)
        self.line2 = self.setup_line("https://xxx", 380, 900, 310, 30, 15)
        self.line3 = self.setup_line("90%", 1100, 503, 50, 30, 12)
        self.line4 = self.setup_line("90%", 1570, 503, 50, 30, 12) 
            
    #检测测试(含视频检测)
    def detect_test(self):
        if(self.detect_permit==0):
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '您的权限不够')
            msg_box.exec_()
            return
        if self.flag_test|self.flag_video_test == 1:
            self.btn2.setText("Testing...")
            self.btn_self(self.btn2, 20, btn_style_text='background-color:yellow')
        else:
            return
        print('load original file from:', self.file_dir)
        file_name = self.file_dir
        detect(file_name)
        new_file_name = self.file_dir[self.file_dir.rfind('/')+1:]
        new_file_name = os.path.join(os.path.abspath(os.getcwd()), 'inference/output/'+new_file_name)
        print('load test file from:',new_file_name)            
        print(self.flag, self.video_flag)
        if self.flag_test == 1 and self.flag_video_test == 0:
            detect(file_name)
            img = QtGui.QPixmap(new_file_name).scaled(self.label2.width(), self.label2.height())
            self.label2.setPixmap(img)
            self.btn2.setText("Success!")
            self.btn_self(self.btn2, 20, btn_style_text='background-color:lightcoral')
        elif self.flag_test == 0 and self.flag_video_test == 1:
            self.slotStart2(new_file_name)
            self.btn2.setText("Success!")
            self.btn_self(self.btn2, 20, btn_style_text='background-color:lightcoral')
            
    
    #分割测试
    def seg_test(self):
        if(self.seg_permit==0):
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '您的权限不够')
            msg_box.exec_()
            return
        if self.file_dir is None:
            return
        else:
            self.flag_seg = 1
        
        if self.flag_seg == 1:
            new_file_name2 = 'train_img/0.jpg'
            seg('data/val/000.jpg')
            
            img2 = QtGui.QPixmap(new_file_name2).scaled(self.label3.width(), self.label3.height())
            self.label3.setPixmap(img2)
            self.btn_seg_test.setText("Success!")
            self.btn_self(self.btn_seg_test, 20, btn_style_text='background-color:lightcoral')
    
    #摄像头检测
    def shell(self):
        if(self.camera_permit==0):
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '您的权限不够')
            msg_box.exec_()
            return
        self.btn2.setText("Testing...")
        self.btn_self(self.btn2, 20, btn_style_text='background-color:yellow')
        file_name = '0'
        detect(file_name)
        self.btn2.setText("Success!")
        self.btn_self(self.btn2, 20, btn_style_text='background-color:lightcoral')
        
            
    #重置窗口
    def reset(self):
        self.file_dir = ''
        self.flag = 0
        self.flag_test = 0
        self.flag_video_test = 0
        self.flag_seg = 0
        self.label.setText("                           insert a picture/video  ")
        self.label2.setText("                           Detection Result  ")
        self.label3.setText("                        Segmentation Result  ")
        self.btn2.setText("Start to Detect")
        self.btn_self(self.btn2, 20, btn_style_text='background-color:lightgreen')
        self.btn_seg_test.setText("Start to Segment")
        self.btn_self(self.btn_seg_test, 20, btn_style_text='background-color:lightgreen')
            
    #文件选择
    def file_select(self):
        
        def open_file():
            file_name = self.file_dir
            img = QtGui.QPixmap(file_name).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(img)
            self.flag = 0
            
        self.openfile_name = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, "选择一个文件", "../", "All(*.*);;Images(*.png *.jpg);;Python文件(*.py)")
        self.file_dir = self.openfile_name[0]
        if '.jpg'  not in self.file_dir:
            pass

        else:
            self.flag = 1
            
        if '.mp4' in self.file_dir:
            self.video_flag = 1
        if self.flag == 1 and self.video_flag == 0:
            open_file()
            img = cv2.imread(self.file_dir)
            path = os.path.join(os.path.abspath(os.getcwd()), 'data/val/000.jpg')
            cv2.imwrite(path, img)
            if self.flag_video_test == 1:
                self.flag_video_test = 0
                self.flag_test = 1
            else:
                self.flag_test = 1
            
        elif self.flag == 0 and self.video_flag == 0:
            self.label.setText("                      the Selected File is not Valid   ")
        elif self.flag == 0 and self.video_flag == 1:
            self.btn_stop1.setHidden(False)
            self.slotStart1()
            if self.flag_test == 1:
                self.flag_test = 0
                self.flag_video_test = 1
            else:
                self.flag_video_test = 1
                    
if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    window = Window()  
    window.show()


    sys.exit(app.exec_())