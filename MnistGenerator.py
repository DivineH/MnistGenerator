import cv2
import time
import numpy as np
import MnistRecognition as mr

Drawing = False
IX, IY = -1, -1

def drawNumber(Event, X, Y, Flags, Param):
    global Img
    R = 255
    G = 255
    B = 255
    Color = (B, G, R);
    global IX, IY, Drawing
    if Event == cv2.EVENT_LBUTTONDOWN:
        Drawing = True
        IX, IY = X, Y
    elif Event == cv2.EVENT_MOUSEMOVE and Flags == cv2.EVENT_FLAG_LBUTTON:
        if True == Drawing:
            cv2.line(Img, (IX, IY), (X, Y), Color, 2)
            IX, IY = X, Y
    elif Event == cv2.EVENT_LBUTTONUP:
        Drawing = False
        smallImg = cv2.resize(Img, (28, 28), interpolation = cv2.INTER_AREA)
        ret, dstImg = cv2.threshold(smallImg, 10, 255, cv2.THRESH_BINARY)
        print("你写的数字是: ", mr.recognition(dstImg))
        cv2.waitKey(1000)
        #imgTitle = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())) + ".jpg"
        #cv2.imwrite(imgTitle, dstImg)
        ret, Img = cv2.threshold(Img, 256, 0, cv2.THRESH_BINARY)
        
Img = np.zeros((560,560,1),np.uint8)
cv2.namedWindow('Image')
#鼠标事件回调函数
cv2.setMouseCallback('Image',drawNumber)

while(1):
    cv2.imshow('Image',Img)
    k = cv2.waitKey(10) & 0xFF
    
    #是否退出
    if k == 27:
        break

cv2.destroyAllWindows()