import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
#list of all imported images from the folder
images = []
#List of image names from the folder
classNames =[]
myList = os.listdir(path)
print(myList)

#cl stands for class
for cl in myList:
    curImg =cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#importing image and converting to RGB
def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Finding the encodings
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

def markAttendance(name):
    #Write down name and time
    with open('Attendance.csv','r+') as f:
        myDataList =f.readlines()
        nameList =[]
        #print(myDataList)
        for line in myDataList:
            entry =line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('No of elements: ' + str(len(encodeListKnown)) + '\nEncoding Complete')

#Find matches between our encodings
cap=cv2.VideoCapture(0)

while True:
    success, img =cap.read()
    #Reduce size of image by 25% to speed up the process
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)

    #Convert to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #Find location of faces as  there may be many faces
    facesCurFrame = face_recognition.face_locations(imgS)

    #Find encoding of the webcam
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    #Finding matches
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown, encodeFace)

        #lowest distance in the list will be best match
        #print(faceDis)

        #Find lowest distance in the list which will be the best match
        matchIndex =np.argmin(faceDis)

        if matches[matchIndex]:
            name =classNames[matchIndex].upper()
            #print(name)

            #printing a frame
            y1,x2,y2,x1=faceLoc
            #restoring original size 25% = 4
            y1, x2, y2, x1= y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,0),2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

        



# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0], faceLoc[1], faceLoc[2]),(255,0,255),2)
# #print(faceLoc)
# 
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0], faceLocTest[1], faceLocTest[2]),(255,0,255),2)
# 
# #Comparison 
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# #True means similar
# print(results)
# 
# #How similar are 2 faces?? Compare face distance -The smaller the distance the better the match is
# faceDis =face_recognition.face_distance([encodeElon],encodeTest)