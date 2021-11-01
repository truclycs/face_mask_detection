import time
import cv2

from BGSubtractor import BGSubtractor


detector = BGSubtractor(threshold=10, minArea=1000, padding=0.2)
scalefactor = 0.75


video = cv2.VideoCapture(0)
# read the file
while (video.isOpened()):
    # read the image in each frame
    success, image = video.read()

    # check if the file has read the end
    if not success:
        break

    start = time.time()
    image = cv2.resize(image, (0, 0), fx=scalefactor, fy=scalefactor)
    movingObjects = detector.detect(image)
    end = time.time()
    # print((end - start)*1000)
    if movingObjects is not None:
        bgcount = 0
        haarcount = 0
        for obj in movingObjects:
            [x, y, w, h] = obj
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bgcount += 1

        print('Time', int((end - start)*1000), "BG :", bgcount, 'HAAR : ', haarcount)
        cv2.imshow('frame', image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
        