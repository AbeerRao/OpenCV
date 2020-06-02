import os
import cv2
from PIL import Image
import numpy as np
import pickle

# The base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The image directory
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
faceCascade = cv2.CascadeClassifier("facecascaed.xml")
sideCascade = cv2.CascadeClassifier("sidecascaed.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
y_labels = []
x_train = []
current_id = 0
label_ids = {}

# Seeing all the images
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # y_labels.append(label)
            # x_train.append(path)
            pilImage = Image.open(path).convert("L") # Converting to grayscale
            imageArray = np.array(pilImage, "uint8")
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.05, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = imageArray[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)


recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")