import cv2
import numpy as np


def drawRec(biggestNew, mainFrame):
    cv2.line(mainFrame, (biggestNew[0][0][0], biggestNew[0][0][1]), (biggestNew[1][0][0], biggestNew[1][0][1]), (0, 255, 0), 20)
    cv2.line(mainFrame, (biggestNew[0][0][0], biggestNew[0][0][1]), (biggestNew[2][0][0], biggestNew[2][0][1]), (0, 255, 0), 20)
    cv2.line(mainFrame, (biggestNew[3][0][0], biggestNew[3][0][1]), (biggestNew[2][0][0], biggestNew[2][0][1]), (0, 255, 0), 20)
    cv2.line(mainFrame, (biggestNew[3][0][0], biggestNew[3][0][1]), (biggestNew[1][0][0], biggestNew[1][0][1]), (0, 255, 0), 20)

img = cv2.imread("IMG-2012.jpg")

#Keeping Actual Resolution
img = cv2.resize(img,(int(480*2), int(640*2)))
w,h = 480,640
WarpImg = img.copy()


#Process
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurredframe = cv2.GaussianBlur(grayimg, (5,5), 1)
CannyImage = cv2.Canny(blurredframe, 190,190)
contours,_ = cv2.findContours(CannyImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursframe = img.copy()
Contoursframe = cv2.drawContours(contoursframe, contours,-1, (255,0,0),4)

corner_frame = img.copy()
maxArea = 0
biggest = []

for i in contours:
    area = cv2.contourArea(i)
    if area > 500:
        peri = cv2.arcLength(i, True) #Closed Loop
        edges = cv2.approxPolyDP(i,0.02*peri, True) #Approx Edges
        if area > maxArea and len(edges)==4:
            biggest = edges
            maxArea = area

if len(biggest)!=0:
    
    
    biggest =biggest.reshape((4,2))
    biggestNew = np.zeros((4,1,2), dtype=np.int32)
    add = biggest.sum(1)
    biggestNew[0] = biggest[np.argmin(add)]
    biggestNew[3] = biggest[np.argmax(add)]
    dif = np.diff(biggest, axis=1)
    biggestNew[1] = biggest[np.argmin(dif)]
    biggestNew[2] = biggest[np.argmax(dif)]
    drawRec(biggestNew,corner_frame)

    corner_frame = cv2.drawContours(corner_frame, biggestNew, -1, (255,0,255),25)

    pts1 = np.float32(biggestNew)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img, matrix,(w,h))
    


#For Display Only
img = cv2.resize(img, (480,640))
grayimg = cv2.resize(grayimg, (480,640))
blurredframe = cv2.resize(blurredframe, (480,640))
CannyImage = cv2.resize(CannyImage, (480,640))
ContoursFrame = cv2.resize(Contoursframe,(480,640))
corner_frame = cv2.resize(corner_frame,(480,640))

cv2.imshow("Image",img)
##cv2.imshow("Gray Image",grayimg)
##cv2.imshow("Gaussian Image",blurredframe)
##cv2.imshow("Canny Image",CannyImage)
##cv2.imshow("Contours",ContoursFrame)
##cv2.imshow("Corner_Frame",corner_frame)
cv2.imshow("Output",imgWarp)
cv2.waitKey(0)
