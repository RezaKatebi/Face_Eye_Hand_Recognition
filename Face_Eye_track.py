#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:13:11 2018

@author: rezakatebi
"""
# Importing opencv 
import cv2 

# Captruing video through your webcam 
video = cv2.VideoCapture(0)
#video.set(3,640)
#video.set(4,480)

###################################################################
# We are using ascades to train classifiers for the object 
# that we are interested 

# The following is a face classifier using the frontal face cascade
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# The following is a eye glasses classifier using the eye cascade
eyceglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# The following is a smile classifier using the hand cascade 
hand_cascade = cv2.CascadeClassifier('hand.xml')
###################################################################
# Let's read the video frame by frame and detect faces and eyes 
while True:
  
  # Reading in the frame
  _,frame = video.read()
  
  # Changing color BGR to gray 
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  # Detecting faces 
  faces = facecascade.detectMultiScale(gray,1.3,5)
  
  # Looping throug each face
  for (x,y,w,h) in faces:
    
    # Drawing rectangle around each face 
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 6)
    
    # Writing face on the top of each face 
    cv2.putText(frame, "Face", (x,y),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    
    # Choosing the area of the image that contains the face
    # both in color and gray 
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    

    # Identifying eyes for each face 
    eyeglasses = eyceglasses_cascade.detectMultiScale(roi_gray)
    
    # Looping through the eyes 
    for (gx,gy,gw,gh) in eyeglasses:
      
      # Drawing rectangle around the eyes and writing eye on them 
      cv2.rectangle(roi_color,(gx,gy), (gx+gw, gy+gh), (0,0,255),2)
      cv2.putText(roi_color, "eye", (gx,gy),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
      
  
  # Now we detect hands in the image   
  hands = hand_cascade.detectMultiScale(gray,1.3,5)
  
  # Looping through hands
  for (sx,sy,sw,sh) in hands:
    
    # Draw rectangle around the hand and call it hand 
    cv2.rectangle(frame, (sx,sy), (sx+sw, sy+sh), (255,255,0),2)
    cv2.putText(frame, "hand", (sx,sy),
              cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
      
      
    
  # Show the frame  
  cv2.imshow('frame', frame)
  
  # Define a stop key (here I defined q)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video and close all the windows  
video.release()
cv2.destroyAllWindows()
        
      
    
    
    