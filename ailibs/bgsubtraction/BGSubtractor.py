"""
    Copyright (C) 2016-2018 Hitachi Asia Ltd. All Rights Reserved.
"""
import os
import sys
import cv2


class BGSubtractionParammeter:
    minArea = 2000
    threshold = 10
    padding = 0.2


class BGSubtractor:
    """
    Object detector using Haar feature and Cascade.

    Attributes:
        __mBackground (numpy array): background image.
        __mThreshold (float): threshold to binary image after subtract background
        __mMinArea (int): minimum area threshold
        __mPadding (float): padding scale for detected moving objects
    """

    def __init__(self , 
                    threshold=BGSubtractionParammeter.threshold,
                    minArea=BGSubtractionParammeter.minArea,
                    padding=BGSubtractionParammeter.padding):
        """
        Constructor.
        Args:
            threshold (float): threshold to binary image after subtract background
            minArea (int): minimum area threshold
            padding (float): padding scale for detected moving objects
        """

        self.__mBackground = None
        self.__mThreshold = threshold
        self.__mMinArea = minArea
        self.__mPadding = padding
        self.__mCount = 0

    def detect(self, 
                npImage):
        """
        Detect objects in given image using loaded model.
        Args:
            image (numpy array): image contains objects.

        Returns:
            moving_objects (list): list of detected objects [x, y, w, h] in image.
        """

        image = npImage.copy()

        moving_objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        if self.__mBackground is None:
            self.__mBackground = gray
            return moving_objects

        if self.__mBackground.shape != gray.shape:
            raise Exception("[BGSubtractionDetector]: Non matching size!")
            return moving_objects

        fgmask = cv2.absdiff(self.__mBackground, gray)
        self.__mBackground = gray
        thresh = cv2.threshold(fgmask, BGSubtractionParammeter.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.__mMinArea:
                continue
     
            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            moving_objects.append([x, y, w, h])

        return moving_objects
