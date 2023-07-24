# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:23:54 2023

@author: Lavanya
"""

import numpy as np
from keras.models import model_from_json
import operator
import cv2 as cv
import sys, os
import pyautogui as p 


json_file = open("gesture-model.json", "r")
model_json = json_file.read()
json_file.close()
load_model = model_from_json(model_json)

load_model.load_weights("gesture-model")
print("Loaded model from disk")

label = ""
action=""

vid = cv.VideoCapture(0)


import webbrowser
from googleapiclient.discovery import build

API_KEY = 'AIzaSyCLeqCz_P-U2H1ELLIQOcZSYvFpjGPpChg'
you_tube = build('youtube', 'v3', developerKey=API_KEY)

request = you_tube.search().list(q='cats', part='snippet', maxResults=1)
response = request.execute()

id = response['items'][0]['id']['videoId']
    

url = f'https://www.youtube.com/watch?v={id}'


webbrowser.open(url)


while (vid.isOpened()):

    ret,frame = vid.read()
    if ret:
            frame = cv.flip(frame, 1)

          
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            
            cv.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0),3)
            
            roi = frame[y1:y2, x1:x2]

          
            roi = cv.resize(roi, (120, 120))
            roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            _, test_image = cv.threshold(roi, 130, 255, cv.THRESH_BINARY)
            cv.imshow("Test Image", test_image)
            res = load_model.predict(test_image.reshape(1, 120, 120, 1))
            pred = {'palm': res[0][0],
                          'fist': res[0][1],
                          'thumbs-up': res[0][2],
                          'thumbs-down': res[0][3],
                          'index-right': res[0][4],
                          'index-left': res[0][5],
                          'no-gesture':res[0][6]}
            
            pred = sorted(pred.items(), key=operator.itemgetter(1), reverse=True)

            if(pred[0][0] == 'palm'):
                label = 'palm'
                action = "PLAY/PAUSE"
                p.press('playpause', presses=1)
            elif (pred[0][0] == 'fist'):
                label = 'fist'
                action = "MUTE"
                p.press('volumemute', presses=1)
            elif (pred[0][0] == 'thumbs-up'):
                label = "thumbs-up"
                action = "VOLUME UP"
                p.press('volumeup', presses=1)
            elif (pred[0][0] == "thumbs-down"):
                label = "thumbs-down"
                action = "VOLUME DOWN"
                p.press('volumedown', presses=1)
            elif (pred[0][0] == "index-right"):
                label = "index-right"
                action = "FORWARD"
                p.press('nexttrack', presses=1)
            elif (pred[0][0] == "index-left"):
                label = "index-left"
                action = "REWIND"
                p.press('prevtrack', presses=1)
            elif (pred[0][0] == "no-gesture"):
                label = "no-gesture"
                action = "NO-ACTION"
            t1= "Gesture: {}".format(label)
            t2= "Action:{}".format(action)

            cv.putText(frame, t1 , (10, 120), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
            cv.putText(frame, t2 , (10, 220), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
            cv.imshow("Hand Gesture Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
vid.release()
cv.destroyAllWindows()