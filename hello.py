from cgitb import reset
from random import sample
from unittest import result
from cv2 import SIFT, DescriptorMatcher, imread
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
kp = None
img = cv2.imread("C:/Users/vanvi/Pictures/thinned.png")
img = cv2.resize(img , None , fx=2, fy=2)
img3 =img
#Threshold:
#ret,img = cv2.threshold(img,195,255,0)


sift= cv2.SIFT_create()
kp, des = sift.detectAndCompute(img,None)
img2=cv2.drawKeypoints(img,kp,img)
cv2.imshow('img',img)
#cv2.imshow('img3',img3)
#plt.hist(img.ravel(),256,[0,256]); plt.show()
#bdo = img.ravel(),256,[0,256]
#y,x = plt.hist(bdo)
#print (x.max)
#print (y.max)
cv2.waitKey(0)
cv2.destroyAllWindows()

