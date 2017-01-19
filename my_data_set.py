import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

def fetch_data():
    img_array =[]
    distance_array = []
    path1 = 'image_set/image1'
    for filename in os.listdir(path1):
        new_path = path1 + '/' +filename
        img = cv2.imread(new_path,0)
        #img = img.resize((300,300))
        img_array.append(img)
        distance_array.append(int(filename[-7:-4]))
    path1 = 'image_set/image2'
    for filename in os.listdir(path1):
        new_path = path1 + '/' +filename
        img = cv2.imread(new_path,0)
        #img = img.resize((300,300))
        img_array.append(img)
        distance_array.append(int(filename[-7:-4]))
    path1 = 'image_set/image3'
    for filename in os.listdir(path1):
        new_path = path1 + '/' +filename
        img = cv2.imread(new_path,0)
        #img = img.resize((300,300))
        img_array.append(img)
        distance_array.append(int(filename[-7:-4]))
    path1 = 'image_set/image4'
    for filename in os.listdir(path1):
        new_path = path1 + '/' +filename
        img = cv2.imread(new_path,0)
        #img = img.resize((300,300))
        img_array.append(img)
        distance_array.append(int(filename[-7:-4]))
    path1 = 'image_set/image6'
    for filename in os.listdir(path1):
        new_path = path1 + '/' +filename
        img = cv2.imread(new_path,0)
        #img = img.resize((300,300))
        img_array.append(img)
        distance_array.append(int(filename[-7:-4]))
    path1 = 'image_set/image8'
    for filename in os.listdir(path1):
        new_path = path1 + '/' +filename
        img = cv2.imread(new_path,0)
        #img = img.resize((300,300))
        img_array.append(img)
        distance_array.append(int(filename[-7:-4]))
    #print img_array
    #print distance_array
    return img_array , distance_array

fetch_data()
