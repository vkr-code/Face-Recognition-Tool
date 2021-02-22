import cv2
import numpy as np
import pandas as pd

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='./Data/'
file_name=input("Enter the name of the person: ")
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key= lambda f:f[2]*f[3])
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
    cv2.imshow("frame",frame)
    cv2.imshow("face_section",face_section)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully saved at "+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()


    
