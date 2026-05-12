# Below is my completed Lab 07 Facial Recognition file, which can be the foundation of our user interface.
# It uses the pre-trained model DeepFace, and I have commented `# TO REPLACE` in areas where we need to use our own fine-tuned model. 
# It refers to the images in the `faces_db/` directory, which can be replaced with the full image set provided for the assignment.
# - Jack


import os
import numpy as np
import pandas as pd
from deepface import DeepFace
import cv2
from pathlib import Path

db_path = Path("datasets/faces_db")
classes = [x.name for x in db_path.iterdir() if x.is_dir()]
classes
on_screen = {}

for c in classes:
    on_screen[c] = False

on_screen
import datetime
import csv

def record_event(person, event):
    now = datetime.datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M")
    
    if event == 'entry':
        print(f"{person} IS HERE! {timestamp}")
    elif event == 'exit':
        print(f"{person} has left. {timestamp}")

    # Write to a CSV file
    if Path('appearance_logs.csv').exists():
        with open('appearance_logs.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([person, event.upper(), timestamp])
    else:
        with open('appearance_logs.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([
                ['Name', 'Event', 'Time'],
                [person, event.upper(), timestamp]
            ])
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    prev_on_screen = on_screen
    on_screen = {p: False for p, x in on_screen.items()}
    
    face_objs = DeepFace.extract_faces(img_path=frame, enforce_detection=False) # TO REPLACE Face detection (identify bounding boxes)
    for face_obj in face_objs:
        box = face_obj['facial_area']
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        if face_obj['confidence'] > 0.8 : # Not equal to zero
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            identity_obj = DeepFace.find(img_path = face_obj['face'], db_path = "datasets/faces_db", silent=True, enforce_detection=False) # TO REPLACE Face recognition (compare the on-screen face with images in the faces_db directory to find the closest match)
            if not identity_obj[0].empty:
                similar_photo = Path(identity_obj[0]['identity'][0])
                person = similar_photo.parent.name # Parent directory name
                on_screen[person] = True
                cv2.putText(frame, person, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    for c in classes:
        if on_screen[c] and not prev_on_screen[c]:
            record_event(c, 'entry')
        elif prev_on_screen[c] and not on_screen[c]:
            record_event(c, 'exit')

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()