""" Copyright (C) 2016-2018 Hitachi Asia Ltd. All Rights Reserved. """
import os
import sys
import cv2

from BGSubtractor import BGSubtractor
# from cvlib.detector.haar.MultiHaarDetector import MultiHaarDetector

# add absolute path of cvlib to PYTHONPATH
PYTHON_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", ".."))
# PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PYTHON_PATH)
os.chdir(PYTHON_PATH)

SKIP_FRAME = 1


class BGSubtractionParammeter:
    minArea = 1000
    threshold = 10
    padding = 0.2


class BGSubtractionDetector:
    """
    Object detector using Haar feature and Cascade.

    Attributes:
        __mBackground (numpy array): background image.
        __mThreshold (float): threshold to binary image after subtract background
        __mMinArea (int): minimum area threshold
        __mPadding (float): padding scale for detected moving objects
        __mDetector (object): detector using Haar-Cascade.
    """

    def __init__(self, objectTypes=[], skipFrame=SKIP_FRAME, vehicleClassifier=0,
                 threshold=BGSubtractionParammeter.threshold, minArea=BGSubtractionParammeter.minArea,
                 padding=BGSubtractionParammeter.padding, gpu=False):
        """
        Constructor.

        Args:
            objectType (int): type of object.
            vehicleClassifierMode (int): vehicle classifier type
            threshold (float): threshold to binary image after subtract background
            minArea (int): minimum area threshold
            padding (float): padding scale for detected moving objects
        """

        self.__mVehicleClassifier = vehicleClassifier
        self.__mSkipFrame = 1
        self.__mBackgroundSubtractor = BGSubtractor(threshold, minArea, padding)
        # self.__mDetector = MultiHaarDetector(self.__mSkipFrame, objectTypes, self.__mVehicleClassifier, gpu)
        self.__mCount = 0

    def __str__(self):
        """ Name of BGSubtractionDetector class. """

        name = self.__class__.__name__ + '_' + str(self.__mDetector)
        return name

    def detect(self, image):
        """
        Detect objects in given image using mix of background subtraction and HAAR/SVM.

        Args:
            image (numpy array): image contains objects.

        Returns:
            results (list): list of detected objects [x, y, w, h] in image.
        """

        results = []
        if self.__mCount % self.__mSkipFrame == 0:
            movingObjects = []
            movingObjects = self.__mBackgroundSubtractor.detect(image)
            if movingObjects is not None:
                results = self.classifyImage(image)
        self.__mCount = (self.__mCount + 1) % self.__mSkipFrame

        return results

    def classifyObjects(self, image, movingObjects):
        """
        Classify moving objects using HAAR/SVM.

        Args:
            image (numpy array): input image
            movingObjects (list): list of moving objects detected by BGSubtraction

        Returns:
            objList (list): list of detected objects [x, y, w, h] in image.
        """

        objList = []
        for pos in movingObjects:
            [x0, y0, w0, h0] = pos
            [x, y, w, h] = [
                max(0, x0 - int(BGSubtractionParammeter.padding*x0)),
                max(0, y0 - int(BGSubtractionParammeter.padding*y0)),
                min(image.shape[1], w0 + int(BGSubtractionParammeter.padding*w0)),
                min(image.shape[0], h0 + int(BGSubtractionParammeter.padding*h0))]

            subImage = image[y:y+h, x:x+w]
            objs = self.__mDetector.detect(subImage)

            if objs:
                for obj in objs:
                    [objType, shape, position] = obj
                    [x1, y1, w1, h1] = position
                    [x1, y1, w1, h1] = [x1+x, y1+y, w1, h1]
                    objList.append([objType, shape, [x1, y1, w1, h1]])

        return objList

    def classifyImage(self, image):
        """
        Classify moving objects using HAAR/SVM.

        Args:
            image (numpy array): input image

        Returns:
            objList (list): list of detected objects [x, y, w, h] in image.
        """

        objList = []
        objList = self.__mDetector.detect(image)
        return objList


if __name__ == '__main__':
    import time
    # objectType = OBJECTTYPE.HUMAN
    # print(cv2.__version__)
    scalefactor = 0.75
    objectTypes = [1, 1, 1]
    detector = BGSubtractionDetector(objectTypes)
    # video_path = '/home/nhanvo/kdeploy/videos/TownCentreXVID.avi'

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
        movingObjects, objList = detector.detect(image)
        end = time.time()
        # print((end - start)*1000)
        if movingObjects is not None:
            bgcount = 0
            haarcount = 0
            for obj in movingObjects:
                [x, y, w, h] = obj
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bgcount += 1
            for obj in objList:
                [objType, shape, positions] = obj
                for pos in positions:
                    [x, y, w, h] = pos
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    image = cv2.putText(image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
                    haarcount += 1

            print('Time', int((end - start)*1000), "BG :", bgcount, 'HAAR : ', haarcount)
            cv2.imshow('frame', image)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break
