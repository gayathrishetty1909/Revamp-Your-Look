##VIRTUAL MAKEUP TRYON

import cv2
import dlib
import numpy as np

##Input Image
#Real Time

#import cv2 as cv
#cam = cv.VideoCapture(0)   
#s, img = cam.read()
#if s:
#    cv.namedWindow("Real-time")
#    cv.imshow("Real-time",img)
#    cv.waitKey(0)
    
#Preparing Image
img=cv2.imread(r'Virtual-Makeup-Try-On-System\Example.jpg')
finalimage = img.copy()
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',gray_img)
cv2.imshow('Original Image',img)
# cv2.imwrite('OriginalImage.jpg',img)
# cv2.imwrite('GrayImage.jpg',gray_img)
#cv2.waitKey(0)

#Face Detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Virtual-Makeup-Try-On-System\shape_predictor_68_face_landmarks.dat")
faces = detector(gray_img)
print(faces)

#Detect Face in Image
landmarkspints = []

#empty func for Trackbar
def empty(a):
    pass

#Color selection
#Make Window for Trackbar
cv2.namedWindow("Virtual_Makeup_Try-On")
cv2.resizeWindow("Virtual_Makeup_Try-On",640, 180)
#Red
cv2.createTrackbar("Red","Virtual_Makeup_Try-On", 0, 255, empty)
#Green
cv2.createTrackbar("Green","Virtual_Makeup_Try-On", 240, 255, empty)
#Blue
cv2.createTrackbar("Blue","Virtual_Makeup_Try-On", 225, 255, empty)

for face in faces:
    #1st top left coordinates
    x1,y1 = face.left(), face.top()
    x2,y2 = face.right(), face.bottom()
    img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0),3)
    cv2.imshow("Face Detected", img)
    # cv2.imwrite("facedetect.jpg",img)

#Face Landmarks
landmarks = predictor(gray_img,face)
# print(landmarks)

for n in range(68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmarkspints.append([x,y])
    cv2.circle(img,(x,y),1,(0,255,0),cv2.FILLED)
    cv2.putText(img,str(n),(x+1,y+10),cv2.FONT_HERSHEY_PLAIN,0.8,(0,0,255),1)
    #Lipstick - lip colour, lip mask - 48-60
     
#crop lip region
landmarkspints = np.array(landmarkspints)
# mask
lipmask = np.zeros_like(img)
lipimg = cv2.fillPoly(lipmask, [landmarkspints[48:60]],(255,255,255))
cv2.imshow("lip",lipimg)  
cv2.imshow("Face Landmarks",img)
# cv2.imwrite("lip.jpg",lipimg)  
# cv2.imwrite("Face Landmarks.jpg",img)


# lip colours
lipimgcolor = np.zeros_like(lipimg)

while True:
    # set colours
    #b = 255
    #g = 1
    #r = 25
    b = cv2.getTrackbarPos("Blue","Virtual_Makeup_Try-On")
    g = cv2.getTrackbarPos("Green","Virtual_Makeup_Try-On")
    r = cv2.getTrackbarPos("Red","Virtual_Makeup_Try-On")
    lipimgcolor[:] = b,g,r
    lipimgcolor = cv2.bitwise_and(lipimg,lipimgcolor)
    #lip edges charpe to blur
    lipimgcolor = cv2.GaussianBlur(lipimgcolor,(5,5),5)
    #add to og image
    finalmakeup = cv2.addWeighted(finalimage,1,lipimgcolor, 0.6,0)
    cv2.imshow("Final",finalmakeup)  
    # cv2.imwrite("Final.jpg",finalmakeup)  

    cv2.waitKey(1)