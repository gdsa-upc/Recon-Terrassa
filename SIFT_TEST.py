import cv2
import os

os.chdir("C:\Users\Albert\UPC\\5Q\GDSA\Recon-Terrassa")

img = cv2.imread('test.jpg')

sift = cv2.SIFT()
kp = sift.detect(img,None)

imgkp=cv2.drawKeypoints(img,kp)
cv2.imwrite('sift_keypoints.jpg',imgkp)

cv2.imshow('test',imgkp)
