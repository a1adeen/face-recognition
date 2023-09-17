# face recognition
import face_recognition
import cv2
import numpy as np
from datetime import datetime
# import csv




video_capture = cv2.VideoCaptureAPIs(0)
print(dir(cv2))

# load only known imags

user_image_1 = face_recognition.load_image_file("self.jpg")
user_1_encoding = face_recognition.face_encodings(user_image_1)[0]
