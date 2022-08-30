import cv2
import numpy as np

img = cv2.imread('cards.jpg')

##pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
##
##print(pts1)
##for x in range(0,4):
##    cv2.circle(img,(int(pts1[x][0]),int(pts1[x][1])),5,(0,0,255),cv2.FILLED)
##
##cv2.imshow("Original",img)
##cv2.waitKey(0)


width,height = 250,350
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))
 
 
cv2.imshow("Image",img)
cv2.imshow("Output",imgOutput)
 
cv2.waitKey(0)
