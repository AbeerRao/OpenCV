import cv2

# Creating a cascade classifier
faceCascade = cv2.CascadeClassifier("facecascaed.xml")

# Reading the image
image = cv2.imread("face.jpeg")

# Reading the image as a grayscale image
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Searching for the coordinates of the face
faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.05, minNeighbors=5)

# Adding the rectangular box
for x, y, w, h in faces:
    image = cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,196), 3)

# Resizing the image
resizedImage = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))

# Showing the image
cv2.imshow("Detection", resizedImage)

# When to close the window
cv2.waitKey(0)

# Closing the window
cv2.destroyAllWindows()