# Import the necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Read the image as a grayscale image
img = cv2.imread("F:/CODE/DB1_B/101_2.tif",0)
img = cv2.resize(img , None , fx=1.5, fy=1.5)
img_df = img
# Threshold the image
#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)
imgth =img
#-----------Thinning----------------
# Step 1: Create an empty skeleton
size = np.size(img) # np.size đếm tổng số phần tử trong ma trận aka tổng số điểm ảnh trong ảnh M.N
skel = np.zeros(img.shape, np.uint8) # tạo một ma trận aka ảnh mới với các phần tử là 0, có shape giống shape img và kiểu dữ liệu uint8 - số nguyên có dấu 8bit (0000 0000- 1111 1111 aka 0-255)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) #getStruct để khởi tạo 1 ma trận với (shape, size); MORPH là tùy chọn cho shape (cross, rect, elipse,...) -> để tạo ra kernel tùy ý

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element) # áp dụng phép biển đổi hình thái học Opening = erode rồi dilate -> giảm nhiễu
    #Step 3: Substract open from the original image
    temp = cv2.subtract(img, open) # trừ ảnh 1 cho ảnh 2 (những điểm ở vị trí của ảnh 2 sẽ được hiển thị ngược lại trong ảnh 1)
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(img, element)
    skel = cv2.bitwise_or(skel,temp) #phần chung giữa ảnh 1 và 2 (phép or)
    img = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        #cv2.imshow("nonzero",img)
        break

#sift
sift = cv2.SIFT_create()
kp, des =sift.detectAndCompute(skel,None)
img2 =cv2.drawKeypoints(skel,kp,skel)
# Displaying the final skeleton
imgth=cv2.bitwise_not(imgth)
skel=cv2.bitwise_not(skel)
cv2.imshow('anhgoc', img_df)
cv2.imshow('otsu', imgth)
cv2.imshow("skel",skel)
print(ret)
print(cv2.countNonZero(img))
#plt.subplot(121),plt.imshow(img_df),plt.title('Original')
#plt.subplot(122),plt.imshow(img2),plt.title('Testing')
#plt.show()
# show histogram
#plt.hist(skel.ravel(),256,[0,256]); plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()