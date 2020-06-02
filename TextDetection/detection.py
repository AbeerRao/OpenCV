import cv2
import pytesseract

# The location of the executable file
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
# The path of the image file

img = cv2.imread("image.jpg")
# Converting the image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Resizing the image
img = cv2.resize(img, (500, 500))

# Printing what characters have been found in the image
#print(pytesseract.image_to_string(img))
# Dtecting the characters in the image
iHeight, iWidth, _ = img.shape
boxes = pytesseract.image_to_boxes(img)
for box in boxes.splitlines():
    box = box.split(" ") # Making each of the character into its own list
    # Getting the coordinates and dimesions of each character as integers
    x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    # Creating a rectangle arpund each character
    cv2.rectangle(img, (x, iHeight-y), (w, iHeight-h), (0, 0, 255), 3)
    # Adding the character which was detected
    cv2.putText(img, box[0], (x, iHeight-y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

# Displaying the image
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
