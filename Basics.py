import cv2
import numpy as np
import face_recognition
#importing image
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
#converting to RGB
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

#importing test image
#imgTest = face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
imgTest = face_recognition.load_image_file('ImagesBasic/Bill gates.jpg')
#converting to RGB
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


#Step2:Finding faces in our image and their encodings
#0 to get first element/value
#Face Loc maps the face into a rectangle
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0], faceLoc[1], faceLoc[2]),(255,0,255),2)
#print(faceLoc)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0], faceLocTest[1], faceLocTest[2]),(255,0,255),2)

#Comparison 
results = face_recognition.compare_faces([encodeElon],encodeTest)
#True means similar
print(results)

#How similar are 2 faces?? Compare face distance -The smaller the distance the better the match is
faceDis =face_recognition.face_distance([encodeElon],encodeTest)
print(faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)


cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)