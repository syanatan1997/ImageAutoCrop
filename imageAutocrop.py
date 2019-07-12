#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:49:13 2019

@author: miyatayuutarou
"""

import os
import cv2
import glob
#assert os.path.isfile("lbpcascade_animeface.xml")

# 特徴量ファイルをもとに分類器を作成
cascadefile = "lbpcascade_animeface.xml"
classifier = cv2.CascadeClassifier(cascadefile)
output_dir = "./CropedImage"
images = [cv2.imread(file) for file in glob.glob("./pixivImages/*.jpg")]
filenum = 0
# 顔の検出
for image in images:
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_image, minSize=(100,100))

    for i, (x,y,w,h) in enumerate(faces):
        # 一人ずつ顔を切り抜く
        face_image = image[y:y+(h+50), x:x+(w+50)]
        scaled = cv2.resize(face_image, dsize=(256,256))
        output_path = os.path.join(output_dir,'{0}.jpg'.format(filenum))
        filenum = filenum + 1
        cv2.imwrite(output_path,scaled)
        #name = i+ '.jpg'
    #cv2.imwrite(i, image)