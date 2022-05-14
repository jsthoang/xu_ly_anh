# Gabor filter in x direction -  Black to white gradient
#Theta - 0 degree 
from ast import Invert
from cgitb import reset
from random import sample
from unittest import result
from cv2 import SIFT, DescriptorMatcher, imread
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("F:\JavaHoc\.vscode\XLA\DB1_B\q101_1.tif")
img = cv2.resize(img, None, fx=2, fy=2)
pi = 3.14
kernel = cv2.getGaborKernel(ksize=(3, 3), sigma=8.0, theta=0, lambd=10.0, gamma=0.0)

gabor = cv2.filter2D(img, cv2.CV_8UC1, kernel)
gabor = cv2.bitwise_not(gabor)
cv2.imshow('img', gabor)
cv2.imshow('img2', img)
cv2.waitKey(0)
cv2.destroyAllWindows()