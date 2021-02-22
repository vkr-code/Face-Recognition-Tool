import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt 
import pandas as pd 

###############KNN CODE############################
def eucledian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def knn(X_train,Y_train,test_point,k=5):
    distances=[]
    for point,label in zip(X_train,Y_train):
        dist=eucledian_distance(point,test_point)
        distances.append((dist,label))
    print(distances)
    distances=sorted(distances,key=lambda x:x[0])
    print("*"*20)
    print(distances)
    distances=np.array(distances)
    print("*"*20)
    print(distances)
    distances=distances[:k,:]
    print("*"*20)
    print(distances)
    freq=np.unique(distances[:,1],return_counts=True)
    print(freq)
    labels,counts=freq
    print(labels)
    print(counts)
    ans=labels[counts.argmax()]
    return ans
#########KNN CODE##################
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='./data/'
labels=[]
class_id=0
names={}
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        print("Loaded "+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
#face_labels=np.concatenate(labels,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

#Testing
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        #out=knn(trainset,face_section.flatten())
        out=knn(face_dataset,face_labels,face_section.flatten())
        pred_name=names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Faces",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
