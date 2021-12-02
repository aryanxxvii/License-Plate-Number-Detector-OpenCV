import cv2
import numpy as np
import pytesseract as pt

PYTESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
IMG_PATH = "resources/test5.jpg"
print(IMG_PATH)
pt.pytesseract.tesseract_cmd = PYTESSERACT_PATH
numCascade = cv2.CascadeClassifier("cascades/haarcascade_russian_plate_number.xml")

def get_plate_number(path):
    img = cv2.imread(path)


    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    plates = numCascade.detectMultiScale(imgBlur, 1.1, 4)

    for (x, y, w, h) in plates:
        kernel = np.ones((1, 1), np.uint8)

        r = int(0.02*img.shape[0])
        s = int(0.02*img.shape[1])

        imgCropped = img[y+r:y+h-r, x+s:x+w-s,:]
        imgDilated = cv2.dilate(imgCropped, kernel, iterations=1)
        imgEroded = cv2.erode(imgDilated, kernel, iterations=1)
        imgFinal = cv2.cvtColor(imgEroded, cv2.COLOR_RGB2GRAY)
        # ignore_, imgFinal = cv2.threshold(imgGray2,70,255,cv2.THRESH_TOZERO)

        text = pt.image_to_string(imgFinal)
        plateNumber = ""
        for i in text:
            if i.isalnum():
                plateNumber += i
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (70, 70, 255), 2)
        cv2.rectangle(img, (x,y-30), (x+w,y), (255, 70, 70) , -1)
        cv2.putText(img, plateNumber, (x,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Plate Number Detected - {}'.format(plateNumber), img)
        cv2.imshow("plate", imgFinal)
    cv2.waitKey(0)

get_plate_number(IMG_PATH)
