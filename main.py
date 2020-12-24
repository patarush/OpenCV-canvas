import cv2
print('done')

frameWidth=640
frameHeight=480
cap=cv2.VideoCapture(0)
cap.set(3,frameHeight)
cap.set(4,frameWidth)
cap.set(10,150)
while True:
    success,img2=cap.read()
    cv2.imshow('video',img2)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break

