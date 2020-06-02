import cv2, time
import numpy as np
import pickle

# Using the built-in camera
video = cv2.VideoCapture(0) # The '0' means that the video will be from the default camera

a = 1

# Creating a cascade classifier and recognizer
faceCascade = cv2.CascadeClassifier("facecascaed.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

recognizer.read("trainer.yml")

# Running a while loop to capture entire video
while True:
    a += 1
    # The check and frame variables
    check, frame = video.read()
    # Converting to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecing the faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=9)
    # Adding the rectangular box
    for (x, y, w, h) in faces:
        regionOIG = gray[y:y+h, x:x+w]
        regionOIC = frame[y:y+h, x:x+w]
        # Recognizing the faces
        id_, conf = recognizer.predict(regionOIG)
        if conf >= 45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 2, color, stroke, cv2.LINE_AA)
        imageItem = "recognized.png"
        cv2.imwrite(imageItem, regionOIC)

        color = (255, 0, 0) # BGR
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)
    # Showing the video
    cv2.imshow("Recognizing...", frame)
    # When to close the close the frame
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Printing the numbe of frames
print(a)

# Closing the video
video.release()

# Closing the window
cv2.destroyAllWindows()