# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import thread
import cv2
import numpy as np
import RPi.GPIO as GPIO
from sklearn import preprocessing
from sklearn.externals import joblib

################# IMPORT TRAINING DATA SOURCE ##########################
filename = 'model0807_94.sav'
clf = joblib.load(filename)
########################################################################
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 40
camera.shutter_speed = 11
rawCapture = PiRGBArray(camera, size=(320, 240))

GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)
GPIO.setup(18,GPIO.OUT)

# allow the camera to warmup
time.sleep(1)

numBean = 0


def shoot_air(pin,delay):
        #time.sleep(delay/1000)
        GPIO.output(pin,GPIO.LOW)
        time.sleep(0.3)
        GPIO.output(pin,GPIO.HIGH)
       
kernel = np.ones((3,3),np.uint8)
# capture frames from the camera
num_image=382*5
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        start= time.time()
        image= frame.array

        image = image[:,40:305]
        image= cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)

        
        im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##
##        
##        im_in = cv2.GaussianBlur(im_in_first1,(9,9),0) ;
##        # Threshold.
##        # Set values equal to or above 220 to 0.
##        # Set values below 220 to 255.
##         
##        th, im_th = cv2.threshold(im_in, 90, 255, cv2.THRESH_BINARY_INV);
        im_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_shadow = 45
        im_th = cv2.inRange(im_hsv,np.array([0,lower_shadow,0]),np.array([255,255,255]))
############
        

        # Copy the thresholded image.
        im_floodfill = im_th.copy()
         
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
         
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
         
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
         
        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        im_out = cv2.morphologyEx(im_out,cv2.MORPH_OPEN,kernel)


        im2, contours, hierarchy = cv2.findContours(im_out, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print (time.time()-start)*1000
        im_con=cv2.drawContours(image, contours, -1, (200,255,60), 1)
        

        noObject=0
        font = cv2.FONT_HERSHEY_SIMPLEX

        status=1
        shift=-30   
##        im_graY = im_th.copy()
        res_normalize = np.zeros([15,15],dtype=np.uint8)

        
        for cnt in contours:

            M = cv2.moments(cnt)
            if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    area = cv2.contourArea(cnt)
                    x,y,w,h = cv2.boundingRect(cnt)
                    
                    
                    if w < 50 and h <50 and w >10 and h>10:

                            im_gray_mask = im_gray[y:y+h,x:x+w]                 

                            im_bin_mask = im_out[y:y+h,x:x+w]

                            res=cv2.bitwise_and(im_gray_mask,im_gray_mask,mask=im_bin_mask)

                            res_normalize = cv2.resize(res,(15,15),interpolation=cv2.INTER_CUBIC)

################# CONVERT OBJECT TO VECTOR FOR CLASSIFIER ########################
                            object_vector=np.concatenate(res_normalize).reshape(1,-1)
                            object_vector=object_vector/255.0
################## CLASSIFIES BY MLPCLASSIFIER ###################################
                            result = clf.predict(object_vector)
################# WRITE SAMPLE FROM OBJECT #######################################
##                            if num_image%5==0:   
##                                    cv2.imwrite('xau' + str(num_image/10) + '.png',res_normalize)
##                            num_image = num_image+1 
##################################################################################
                            cv2.circle(im_con,(cx,cy),2,(0,0,255),-1)

                            if(result>0.5):

                               cv2.rectangle(im_con,(x,y),(x+w,y+h),(0,255,0),1) 

                            else:
                               cv2.rectangle(im_con,(x,y),(x+w,y+h),(0,0,255),1)      
##
##                            cv2.putText(im_con,'area: ' + str(area),(cx-50,cy-50-shift), font, 0.2,(0,255,128),1,cv2.LINE_AA)
##                            cv2.putText(im_con,'moment: ' + str(cx) +',' + str(cy),(cx-50,cy-40-shift), font, 0.2,(0,255,128),1,cv2.LINE_AA)
##
##                            if status:
##                                cv2.putText(im_con,'status: Good' ,(cx-50,cy-30-shift), font, 0.2,(0,255,128),1,cv2.LINE_AA)
##                            else:
##                                cv2.putText(im_con,'status: Bad' ,(cx-50,cy-30-shift), font, 0.2,(0,255,128),1,cv2.LINE_AA)
##

##                            thread.start_new_thread(shoot_air,(18,cy))
                            
                            
                            noObject=noObject+1
##                            numBean +=1
                        
                


            
                
        cv2.putText(im_con,'No. Object ' + str(noObject),(0,20), font, 0.2,(0,255,128),1,cv2.LINE_AA)

##        cv2.imwrite('capturue' + str(numBean) + '.png',image)
        cv2.namedWindow('Frame',0)
        cv2.resizeWindow('Frame',500,500)
        cv2.imshow("Frame", im_con)

        cv2.namedWindow('Frame1',0)
        cv2.imshow("Frame1", res_normalize)
        # show the frame
        key = cv2.waitKey(1) & 0xFF


##        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
##        if key == ord("q"):
##                break
