import cv2
import numpy as np
print('imported')

"""
img=cv2.imread('resources/sample.jpg')
cv2.imshow('output',img)
cv2.waitKey(0)
"""
#playing video

"""
cap=cv2.VideoCapture('resources/sample-mp4-file.mp4')

while True:
    success,img1=cap.read()
    cv2.imshow('video',img1)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
"""
#accessing webcam

"""
cap=cv2.VideoCapture(0)
cap.set(3,853)
cap.set(4,1280)
cap.set(10,100)
while True:
    success,img2=cap.read()
    cv2.imshow('video',img2)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
"""
########################### chapter 2 ######################

"""
kernel=np.ones((5,5),np.uint8)

img3=cv2.imread('resources/sample.jpg')
imggray=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
imgblur=cv2.GaussianBlur(imggray,(7,7),0)#(7,7)has to be odd
imgcanny=cv2.Canny(img3,150,200) #increasing 150,100 decreases edges
imgdialation=cv2.dilate(imgcanny,kernel,iterations=1) #range of iteration 1-5
imgeroded=cv2.erode(imgdialation,kernel,iterations=1)

cv2.imshow('GRAY',imggray)
cv2.imshow('BLUR',imgblur)
cv2.imshow('CANNY',imgcanny)
cv2.imshow('DIALATION',imgdialation)
cv2.imshow('ERODED',imgeroded)
cv2.waitKey(0)
"""

########################### chapter 3 ######################

# x = (0,0)--->
# y =     |
#         v
"""
img4=cv2.imread('resources/sample.jpg')


print(img4.shape)
imgResize=cv2.resize(img4,(623,462)) #width-height
imgCropped=img4[0:462,0:500] #Height-width
cv2.imshow('Orginal',img4)
cv2.imshow('Resized',imgResize)
cv2.imshow('Cropped',imgCropped)
cv2.waitKey(0)
"""
########################### chapter 4 ######################

"""
img=np.zeros((512,512,3),np.uint8)
print(img)

#img[:]= 255,0,0

cv2.line(img,(0,0),(300,300),(0,255,0),3)
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)

cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED)

cv2.circle(img,(400,50),30,(255,255,0),1)
cv2.putText(img,'opencv',(300,200),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,150,0),3)
cv2.imshow('image',img)
cv2.waitKey(0)
"""

########################### chapter 5 ######################

"""
img5=cv2.imread('resources/card-ui-design-principles-examples.png')
#print(img5.shape)
#cv2.imshow('CARD',img5)
#cv2.waitKey(0)

width,height = 250,350
pts1=np.float32([[404,79],[627,79],[403,473],[636,476]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)

imgoutput=cv2.warpPerspective(img5,matrix,(width,height))

cv2.imshow('warp',imgoutput)
cv2.imshow('CARD',img5)
cv2.waitKey(0)
"""

########################### chapter 6 ######################
"""
img5=cv2.imread('resources/card-ui-design-principles-examples.png')
imgHor=np.hstack((img5,img5))
imgVer=np.vstack((img5,img5))

cv2.imshow('HoRiZoNTAL',imgHor)
cv2.imshow('VERTICAL',imgVer)
cv2.waitKey(0)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

img5=cv2.imread('resources/card-ui-design-principles-examples.png')

imgstach= stackImages(0.5,([img5,img5,img5],[img5,img5,img5]))
cv2.imshow('stack',imgstach)
cv2.waitKey(0)
"""
########################### chapter 7 ######################
"""
def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
"""
"""
PATH='resources/card-ui-design-principles-examples.png'
cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars',640,240)

cv2.createTrackbar('HUE min','TrackBars',100,179,empty)
cv2.createTrackbar('HUE max','TrackBars',173,179,empty)
cv2.createTrackbar('SAT min','TrackBars',53,255,empty)
cv2.createTrackbar('SAT max','TrackBars',255,255,empty)
cv2.createTrackbar('VAL min','TrackBars',0,255,empty)
cv2.createTrackbar('VAL max','TrackBars',255,255,empty)

while True:
    img6=cv2.imread(PATH)
    imgHsv=cv2.cvtColor(img6,cv2.COLOR_BGR2HSV)

    h_min=cv2.getTrackbarPos('HUE min','TrackBars')
    h_max=cv2.getTrackbarPos('HUE max','TrackBars')
    s_min=cv2.getTrackbarPos('SAT min','TrackBars')
    s_max=cv2.getTrackbarPos('SAT max','TrackBars')
    v_min=cv2.getTrackbarPos('VAL min','TrackBars')
    v_max=cv2.getTrackbarPos('VAL max','TrackBars')
    print(h_min,h_max,s_min,s_max,v_min,v_max)

    lower = np.array([h_min,s_min,v_min])
    upper= np.array([h_max,s_max,v_max])

    mask=cv2.inRange(imgHsv,lower,upper)
    #keep color you want to detect white and which you dont want black

    imgResult=cv2.bitwise_and(img6,img6,mask=mask)


    #cv2.imshow('HSV',imgHsv)
    #cv2.imshow('ORIGINAL',img6)
    #cv2.imshow('MASK',mask)
    #cv2.imshow('IMGRESULT',imgResult)
    #cv2.waitKey(1)

    imgStack=stackImages(0.6,([img6,imgHsv],[mask,imgResult]))
    cv2.imshow('stackRESULT', imgStack)
    cv2.waitKey(1)

"""
########################### chapter 8 ######################
"""
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"



            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)




path = 'Resources/shapes.png'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)
getContours(imgCanny)

imgBlank = np.zeros_like(img)
imgStack = stackImages(0.8,([img,imgGray,imgBlur],
                            [imgCanny,imgContour,imgBlank]))

cv2.imshow("Stack", imgStack)

cv2.waitKey(0)
"""
########################### chapter 9 ######################
"""
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
img = cv2.imread('resources/lena.png')
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('Result',img)
cv2.waitKey(0)
"""
