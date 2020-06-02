import cv2
import numpy as np

##################################################
webCamFeed = True
pathImage = "C:\\Users\\Abeer\\Desktop\\OpenCV\\Scanner\\1.jpg"
cap = cv2.VideoCapture(1)
cap.set(10, 160)
heightImg = 640
widthImg = 480
##################################################

# TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth+len(
                    lables[d][c])*13+27, 30+eachImgHeight*d), (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth*c+10, eachImgHeight *
                                                d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def nothing(x):
    pass


def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src
#######################################################################################################

initializeTrackbars()
count = 0
#######################################################################################################

while True:
    # Blank Image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    #######################################################################################################
    img = cv2.resize(img, (widthImg, heightImg))  # Resizing the image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale Image
    imgBlur = cv2.gaussblur(imgGray, (5, 5), 1)  # Adding gaussblur
    threshold = vaTrackbars()  # The trackbar values for the threshold
    # Applying canny blur and edge detector
    imgThreshold = cv2.Canny(imgBlur, threshold[0], threshold[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    #######################################################################################################
    # All contours
    imgContours = img.copy()  # Copying the origional image
    imgBigConts = img.copy()
    contours, heirarchy = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Finding all the contours
    # Drawing all the contours that were detected
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    #######################################################################################################
    # Biggest contour
    biggest, maxArea = biggestContour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigConts, biggest, -1, (0, 255, 0), 20)
        imgBigConts = drawRectangle(imgBigConts, biggest, 2)
        pts1 = np.float32(biggest)  # Preparing the points for wrap
        pts2 = np.float32(
            [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(
            img, matrix, (widthImg, heightImg))  # Applying the warp
    #######################################################################################################
        # Removing 20PX from each side
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] -
                                        20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
    #######################################################################################################
        # Adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThreshold = cv2.adaptiveThreshold(
            imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThreshold = cv2.bitwise_not(
            imgAdaptiveThreshold)  # Revesing the binary image
        imgAdaptiveThreshold = cv2.medianBlur(
            imgAdaptiveThreshold, 3)  # Removing the noise
    #######################################################################################################
        # The array for diaplaying the things
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigConts, imgWarpColored, imgWarpGray, imgAdaptiveThreshold])
    else:
        imgArray = ([img, imgGray, imgThreshold, imgContours],
                    [imgBlank, imgBlank, imgBlank, imgBlank])
    #######################################################################################################
    # The labels for display
    labels = [["Origional", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Result", stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
