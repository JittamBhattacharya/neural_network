import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image_set/image1/image1_150.jpg',0)
plt.figure
plt.imshow(img , cmap = plt.cm.gray)
plt.show()

print img.shape

res = cv2.resize(img,None,fx= 0.5, fy= 0.5, interpolation = cv2.INTER_CUBIC)
plt.figure
plt.imshow(res , cmap = plt.cm.gray)
plt.show()

print res.shape
