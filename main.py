#pip intall opencv-contrib-python
import os
import cv2 as cv
import numpy as np
people=["Ben Afflek","Elton John","Jerry Seinfield","Madonna","Mindy Kaling"]
#p=[]
#for i in os.listdir(r"C:\Users\ahzam\PycharmProjects\Face -Detector\train"):
    #p.append(i)
#print(p

DIR=r"C:\Users\ahzam\PycharmProjects\Face -Detector\train"
harr_cascade=cv.CascadeClassifier("haar_face.xml")
features=[]
labels=[]
def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print("Training Done --------------------")
features=np.array(features,dtype="object")
labels=np.array(labels)
#print(f"length of the feautres={len(features)}")
#print(f"length of the labels={len(labels)}")
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
face_recognizer.save("face_trained.yml")
np.save("features.npy",features)
np.save("labels.npy",labels)

