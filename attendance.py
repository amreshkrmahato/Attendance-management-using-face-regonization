import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#from PIL import ImageGrab

path1='picturesAttendance'
pictures=[]
classNames= []
my_List=os.listdir(path1)
print(my_List)
for cl in my_List:
    curImg=cv2.imread(f'{path1}/{cl}')
    pictures.append(curImg)
    classNames.append(os.path1.splitext(cl)[0])
print(classNames)


def findEncodings(pictures):
    encodes_List= []
    for img in pictures:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodes=face_recognition.face_encodings(img)[0]
        encodes_List.append(encodes)
    return encodes_List


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        my_Data_List=f.readlines()
        name_List=[]
        for line in my_Data_List:
            entry=line.split(',')
            name_List.append(entry[0])
        if name not in name_List:
            now=datetime.now()
            dt_String=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt_String}')


####FOR CAPTURING-SCREEN RATHER THAN WEBCAM####
#def capture-Screen(bbox=(300,300,690+300,530+300)):
#capScr=np.array(ImageGrab.grab(bbox))
#capScr=cv2.cvtColor(capScr,cv2.COLOR_RGB2BGR)
#return capScr

encodes_ListKnown=findEncodings(pictures)
print('Encoding Complete')

cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    # img=captureScreen()
    imagss=cv2.resize(img,(0,0),None, 0.25, 0.25)
    imagss=cv2.cvtColor(imagss,cv2.COLOR_BGR2RGB)

    faces_Cur_Frame=face_recognition.face_locations(imagss)
    encodessCurFrame=face_recognition.face_encodings(imagss,faces_Cur_Frame)

    for encodesFace,faceLoc in zip(encodessCurFrame,faces_Cur_Frame):
        match_es=face_recognition.compare_faces(encodes_ListKnown, encodesFace)
        face_Dis=face_recognition.face_distance(encodes_ListKnown, encodesFace)
        # print(face_Dis)
        matchIndex=np.argmin(face_Dis)

        if match_es[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            b1,a2,b2,a1=faceLoc
            b1,a2,b2,a1= b1*4,a2*4,b2*4,a1*4
            cv2.rectangle(img,(a1,b1),(a2,b2),(0,255,0), 2)
            cv2.rectangle(img,(a1,b2 - 35),(a2,b2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(a1+6,b2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
