import cv2
import os
from matplotlib import pyplot as plt

#S'indica el directori de treball

os.chdir("C:\Users\Albert\Google Drive\Recon Terrassa")

#Es carga la imatge
img = cv2.imread('test.jpg')

img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

orb = cv2.ORB()
kp = orb.detect(img,None)

kp, des = orb.compute(img, kp)

img2 = cv2.drawKeypoints(img,kp,color=(255,0,255), flags=0)
plt.imshow(img2),plt.show()