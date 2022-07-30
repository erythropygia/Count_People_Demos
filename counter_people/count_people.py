import numpy as np
import cv2 
from PIL import Image

cap = cv2.VideoCapture('video/TestVideo.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

contours_previous = []
contours_now = []
line=300




while (cap.isOpened()):
    ret,frame = cap.read()
    fgmask=fgbg.apply(frame)
    
    
    try:
        thresh=cv2.dilate(fgmask,None,iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        

        for c in cnts:
            if cv2.contourArea(c) < 1000:
                continue
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            contours_now.append([x,y])


        if(len(contours_previous) == 0 ):
            contours_previous=contours_now
            continue
        
        closest_contour_list = []

        for i in range (len(contours_now)):
            minimum=1000000

            for j in range (len(contours_previous)):
                diff_x = contours_now[i][0] - contours_previous[j][0]
                diff_y = contours_now[i][1] - contours_previous[j][1]
                distance = diff_x**2 + diff_y**2
            
                if (distance < minimum):
                    minimum = distance
                    closest_contour = j
            closest_contour_list.append(closest_contour)

        peopleout=0
        peoplein=0

        for i in range(len(contours_now)):
            y_previous = contours_previous[closest_contour_list[i]][1]
            if((contours_now[i][1] < line) and (y_previous > line)):
                peopleout=peopleout+1
        
            if((contours_now[i][1] > line) and (y_previous < line)):
                peoplein=peoplein+1
            
        print("Contours Now: " + str(contours_now[i][1]))
        print("Y Previous: " + str(y_previous))
        contours_previous = contours_now

        cv2.line(frame, (0, line), (frame.shape[1], line),(0,255,255), 2)

        cv2.putText(frame,"In:" + str(peoplein), (10,50),cv2.FONT_HERSHEY_COMPLEX,fontScale=1.0,color=(0,0,0),thickness=2)
        cv2.putText(frame,"Out:" + str(peopleout), (10,100),cv2.FONT_HERSHEY_COMPLEX,fontScale=1.0,color=(0,0,0),thickness=2)

        cv2.imshow('Frame', frame)
        cv2.resizeWindow(frame, 300, 700)
  
        

        #cv2.imshow('FgMASK', fgmask)
        

    except Exception as e:
        print(e)
        break
    
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

