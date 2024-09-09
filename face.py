import face_recognition
import cv2  #it will take the input from the webcam and give it to the face_recognition
import numpy as np
import csv
import os
import glob
from datetime import datetime

video_capture = cv2.VideoCapture(0)

jobs_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\steve_jobs2.webp")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\Ratan_Tata_photo.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

musk_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\musk.jpg")
musk_encoding = face_recognition.face_encodings(musk_image)[0]

Hritik_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\Hritik.jpg")
Hritik_encoding = face_recognition.face_encodings(Hritik_image)[0]

Nimit_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\Nimit.jpg")
Nimit_encoding = face_recognition.face_encodings(Nimit_image)[0]

Ayush_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\Ayush.jpg")
Ayush_encoding = face_recognition.face_encodings(Ayush_image)[0]

Ansh_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\Ansh.jpg")
Ansh_encoding = face_recognition.face_encodings(Ansh_image)[0]

Archi_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\Archi.jpg")
Archi_encoding = face_recognition.face_encodings(Archi_image)[0]

diya_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\diya.jpg")
diya_encoding = face_recognition.face_encodings(diya_image)[0]

Bhavya_image = face_recognition.load_image_file("E:\\SIH\\Facial_rec\\photo\\Bhavya.jpg")
Bhavya_encoding = face_recognition.face_encodings(Bhavya_image)[0]

known_face_encoding = [
    jobs_encoding,
    ratan_tata_encoding,
    musk_encoding,
    Hritik_encoding,
    Nimit_encoding,
    Ayush_encoding,
    Ansh_encoding,
    Archi_encoding,
    diya_encoding,
    Bhavya_encoding
]

known_face_names = [
    "jobs",
    "ratan data",
    "tesla",
    "Hritik",
    "Nimit",
    "Ayush",
    "Ansh",
    "Archisman",
    "Diya",
    "Bhavya"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
inwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rbg_small_frame = small_frame[:,:,:: 1]
    if s:
        face_locations = face_recognition.face_locations(rbg_small_frame)
        face_encodings = face_recognition.face_encodings(rbg_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            names=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                names = known_face_names[best_match_index]

            face_names.append(names)
            if names in known_face_names:
                if names in students:
                    students.remove(names)
                    print(students)
                    current_date = now.strftime("%Y-%m-%d")
                    current_time = now.strftime("%H-%M-%S")
                    inwriter.writerow([names,current_time,current_date])
    
    cv2.imshow("attendance system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

//DIVYAM KHANNA ADDED THIS
