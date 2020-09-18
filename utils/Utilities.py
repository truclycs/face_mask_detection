import csv
import os
import glob
import cv2
import dlib
# import math
import numpy
import yaml
import threading
import operator
from collections import Counter
# import cvlib.utils
import time
import base64

from ast import literal_eval
# from scipy.spatial import distance
# from PIL import Image, ImageDraw, ImageFont

IMAGE_SHOW = 0


class CyclicTimer:
    """
    Cyclic timer, use for ingesting process.
    
    Attributes:
        callback (function): function to callback.
        kwargs (dictionary): arguments of callback function.
        interval (int): timer interval.
        isStart (bool): is start flag.
        lock (object): threading lock object.
        cond (object): threading condition object.
        thread (object): thread object.
    """

    def __init__(self, callback, interval, **kwargs):
        """
        Constructor.

        Args:
            callback (function): input callback function.
            interval (int): initial timer interval.
            kwargs (dictionary): input arguments of callback function.
        """

        self.kwargs = kwargs
        self.callback = callback
        self.interval = interval

        self.isStart = False
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

        # Declare a thread to handle callback
        self.thread = threading.Thread(target = self._callback)
        self.thread.start()

    def _callback(self):
        """
        Callback thread of timer.
        
        Args:
            None
        Returns:
            None
        """

        while (1):
            self.lock.acquire()
            if (self.isStart == False):
                self.cond.wait()
            self.lock.release()
            time.sleep(self.interval)
            self.callback(**self.kwargs)

    def start(self):
        """
        Start cyclic timer.

        Args:
            None
        Returns:
            None
        """

        self.lock.acquire()
        if (self.isStart == True):
            self.lock.release()
            return

        self.isStart = True
        self.cond.notify(1)
        self.lock.release()

    def stop(self):
        """
        Stop cyclic timer.

        Args:
            None.
        Returns:
            None.
        """

        self.lock.acquire()
        self.isStart = False
        self.lock.release()


class Utilities:
    """
    Utility functions.
    """
    MIN_SIZE = (70,70)
    HISTOGRAM_SIZE_HSV = [6, 8, 4]
    HISTOGRAM_SIZE_BGR = [8, 8, 8]

    COLOR_TABLE_BGR = [
        numpy.array([0, 0, 0], dtype=numpy.float32), # BLACK
        numpy.array([255, 255, 255], dtype=numpy.float32), # WHITE
        numpy.array([0, 0, 255], dtype=numpy.float32), # RED
        numpy.array([0, 255, 0], dtype=numpy.float32), # LIME
        numpy.array([255, 0, 0], dtype=numpy.float32), # BLUE
        numpy.array([0, 255, 255], dtype=numpy.float32), # YELLOW
        numpy.array([255, 255, 0], dtype=numpy.float32), # CYAN
        numpy.array([255, 0, 255], dtype=numpy.float32), # MAGENTA
        numpy.array([192, 192, 192], dtype=numpy.float32), # SILVER
        numpy.array([128, 128, 128], dtype=numpy.float32), # GRAY
        numpy.array([0, 0, 128], dtype=numpy.float32), # MAROON
        numpy.array([0, 128, 128], dtype=numpy.float32), # OLIVE
        numpy.array([0, 128, 0], dtype=numpy.float32), # GREEN
        numpy.array([128, 0, 128], dtype=numpy.float32), # PURPLE
        numpy.array([128, 128, 0], dtype=numpy.float32), # TEAL
        numpy.array([128, 0, 0], dtype=numpy.float32), # NAVY
    ]

    COLOR_TABLE_STR = {
        0: "BLACK",
        1: "WHITE",
        2: "RED",
        3: "LIME",
        4: "BLUE",
        5: "YELLOW",
        6: "CYAN",
        7: "MAGENTA",
        8: "SILVER",
        9: "GRAY",
        10: "MAROON",
        11: "OLIVE",
        12: "GREEN",
        13: "PURPLE",
        14: "TEAL",
        15: "NAVY"
    }

    COLOR_TABLE_HSV = [
        # T.B.D need to define HSV table
    ]

    @staticmethod
    def convertRectangle(rect, inputType="dlib"):
        """
        Convert rectangle [x, y, w, h] to [tl.x, tl.y, br.x, br.y] or vice versa.

        Args:
            rect (array): array of rectangle data.
            inputType (string): rectangle type, can be opencv rectangle or dlib rectangle.
        Returns:
            rect (array): converted rectangle.
        """

        if inputType == "cv2":
            return dlib.rectangle(
                int(rect[0]), int(rect[1]),
                int(rect[2]+rect[0]), int(rect[3]+rect[1]))
        else:
            return [ \
                int(rect.left()), int(rect.top()), \
                int(rect.right()-rect.left()), int(rect.bottom()-rect.top())]

    @staticmethod
    def uniteRectangles(a, b, inputType="cv2"):
        """
        Get union of 2 rectangles.

        Args:
            a (array): array of rectangle a data.
            b (array): array of rectangle b data.
            inputType (string): rectangle type, can be opencv rectangle or dlib rectangle.
        Returns:
            rect (array): united rectangle.
        """

        if inputType == "dlib":
            return a+b
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return [x, y, w, h]

    @staticmethod
    def intersectRectangles(a, b, inputType="cv2"):
        """
        Get intersection of 2 opencv's rectangles.

        Args:
            a (array): array of rectangle a data.
            b (array): array of rectangle b data.
            inputType (string): rectangle type, can be opencv rectangle or dlib rectangle.
        Returns:
            rect (array): intersect rectangle.
        """

        if inputType == "dlib":
            return a.intersect(b)
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0:
            return [0, 0, 0, 0]
        return [x, y, w, h]

    @staticmethod
    def convertBase64toImage(imageData):
        """Convert base 64 to image.
        Args:
            imageData (str): base 64 image
        Returns:
            image (numy array) : return image
        """

        imageData = imageData.replace("'", '"')[1:-1]
        imageData += "="*(4-len(imageData)%4)
        image = base64.b64decode(imageData)

        image = numpy.fromstring(image, numpy.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def areaRectangle(r, inputType="cv2"):
        """
        Get area of rectangle.

        Args:
            r (array): array of rectangle data.
            inputType (string): rectangle type, can be opencv rectangle or dlib rectangle.
        Returns:
            area (float): area of rectangle.
        """

        if inputType == "dlib":
            return float(r.area())
        return float(r[2]*r[3])

    @staticmethod
    def centerRectangle(r, inputType="cv2"):
        """
        Get center point coordinates of rectangle.

        Args:
            r (array): array of rectangle data.
            inputType (string): rectangle type, can be opencv rectangle or dlib rectangle.
        Returns:
            center (tuple): center point coordinates of rectangle
        """

        if inputType == "dlib":
            r = Utilities.convertRectangle(r)
        return [int(r[0]+r[2]/2), int(r[1]+r[3]/2)]

    @staticmethod
    def floorRectangle(r, inputType="cv2"):
        """
        Get floor point coordinates of rectangle.

        Args:
            r (array): array of rectangle data.
            inputType (string): rectangle type, can be opencv rectangle or dlib rectangle.
        Returns:
            floor (tuple): floor point coordinates of rectangle
        """

        if inputType == "dlib":
            r = Utilities.convertRectangle(r)
        return [int(r[0]+r[2]/2), int(r[1]+r[3])]

    @staticmethod
    def ceilRectangle(r, inputType="cv2"):
        """
        Get ceil  point coordinates of rectangle.

        Args:
            r (array): array of rectangle data.
            inputType (string): rectangle type, can be opencv rectangle or dlib rectangle.
        Returns:
            ceil (tuple): ceil point coordinates of rectangle
        """

        if inputType == "dlib":
            r = Utilities.convertRectangle(r)
        return [int(r[0]+r[2]/2), int(r[1])]

    @staticmethod
    def computeHistogram(image, color="bgr", mask=None, size=None):
        """
        Calculate histogram.

        Args:
            image (numpy array): image.
            color (string): color space type.
            mask (numpy array): mask image.
            size (tuple): size of image.
        Returns:
            histogram (vector): histogram value of image.
        """

        histogram = None
        if mask is None:
            # generate ellipse mask
            mask = numpy.zeros(image.shape[:2], numpy.uint8)
            cv2.ellipse(
                mask, (int(image.shape[1]/2), int(image.shape[0]/2)),
                (int(image.shape[1]/2), int(image.shape[0]/2)), 0, 0, 360,
                256, -1)

        if color != "hsv":
            if size is None:
                size = Utilities.HISTOGRAM_SIZE_BGR
            histogram = cv2.calcHist(
                [image], [0, 1, 2], mask, size, [0, 256, 0, 256, 0, 256])
        else:
            if size is None:
                size = Utilities.HISTOGRAM_SIZE_HSV
            histogram = cv2.calcHist(
                [image], [0, 1, 2], mask, size, [0, 180, 0, 256, 0, 256])
        return cv2.normalize(histogram, histogram).flatten()

    @staticmethod
    def getMainColors(histogram, number=2, colorSystem="bgr"):
        """
        Get main colors.

        Args:
            histogram (vector): histogram value of image.
            number (int): number of color table.
            colorSystem (string): color space type.
        Returns:
            color (array): appearance color of image.
        """

        colorTable = Utilities.COLOR_TABLE_BGR
        if colorSystem == "hsv":
            colorTable = Utilities.COLOR_TABLE_HSV

        if number > len(colorTable):
            number = 2

        sortingColors = []
        for color in colorTable:
            sortingColors.append(
                Utilities.getColorValueInHistogram(color, histogram, colorSystem))

        return numpy.argsort(numpy.array(sortingColors)).tolist()[::-1][:number]

    @staticmethod
    def getColorValueInHistogram(color, histogram, colorSystem="bgr", diffBins=1):
        """
        Get color value in histogram.

        Args:
            color (tuple): color picker.
            histogram (vector): histogram value of image.
            colorSystem (string): color space type.
            diffBins (int): difference bins.
        Returns:
            sumValue (float): color value.
        """

        # use exact value of color, divide to bins of histogram size
        # to get the color bin value
        size = Utilities.HISTOGRAM_SIZE_BGR
        rangeBin = [256/s for s in size]
        if colorSystem == "hsv":
            size = Utilities.HISTOGRAM_SIZE_HSV
            rangeBin = [180/size[0], 256/size[1], 256/size[2]]
        colorBin = [int(color[i]/rangeBin[i]) for i in range(3)]
        sumValue = 0.0

        # based on range of different bins, select bins which has the same color
        for ibin in range(
            max(0, colorBin[0]-diffBins),
            min(size[0], colorBin[0]+diffBins+1)):
            # color bins near first channel
            for jbin in range(
                max(0, colorBin[1]-diffBins),
                min(size[1], colorBin[1]+diffBins+1)):
                for kbin in range(
                    max(0, colorBin[2]-diffBins),
                    min(size[2], colorBin[2]+diffBins+1)):
                    # color bins near third channel
                    sumValue += histogram[(ibin*size[1]+jbin)*size[2]+kbin]
        # return the frequency value of the index bin in histogram
        return sumValue

    @staticmethod
    def compareHistograms(
        histogram1, histogram2,
        algorithm=cv2.HISTCMP_BHATTACHARYYA):
        """
        Compare histograms.

        Args:
            histogram1 (vector): histogram value of image 1.
            histogram2 (vector):histogram value of image 2.
            algorithm (int): constant of opencv to assign opencv algorithm.
        Returns:
            confidence (float): compare value of two histogram.
        """

        # cv2.HISTCMP_CORREL = 0,
        # cv2.HISTCMP_CHISQR = 1,
        # cv2.HISTCMP_INTERSECT = 2,
        # cv2.HISTCMP_BHATTACHARYYA = 3,
        # cv2.HISTCMP_HELLINGER = HISTCMP_BHATTACHARYYA,
        # cv2.HISTCMP_CHISQR_ALT = 4,
        # cv2.HISTCMP_KL_DIV = 5
        histogram1 = histogram1.astype(numpy.float32)
        histogram2 = histogram2.astype(numpy.float32)
        if histogram1.shape != histogram2.shape:
            return 1.0
        return cv2.compareHist(histogram1, histogram2, algorithm)

    @staticmethod
    def compareHistogramImages(
        image1, image2,
        algorithm=cv2.HISTCMP_BHATTACHARYYA, color="bgr",
        mask1=None, mask2=None, size=None):
        """
        Compare histogram images.

        Args:
            image1 (numpy array): image 1.
            image2 (numpy array): image 2.
            algorithm (int): constant of opencv to assign opencv algorithm.
            color (string): color space type.
            mask1 (numpy array): mask image 1.
            mask2 (numpy array): mask image 2.
            size (tuple): image size.
        Returns:
            confidence (float): compare value of two images.
        """

        histogram1 = Utilities.computeHistogram(image1, color, mask1, size)
        histogram2 = Utilities.computeHistogram(image1, color, mask2, size)
        return cv2.compareHist(histogram1, histogram2, algorithm)

    @staticmethod
    def cropImage(image, rect):
        """
        Crop image to rectangle area.

        Args:
            image (numpy array): image to crop.
            rect (list): rectangle.
        Returns:
            cropImage (numpy array): crop image.
        """
        return image[ \
            max(0, rect[1]):min(rect[1]+rect[3], image.shape[0]), \
            max(0, rect[0]):min(rect[0]+rect[2], image.shape[1])
            ].copy()

    @staticmethod
    def convertListArrayToTupleInt(a):
        """
        Convert a list of an array to integer tuple.

        Args:
            a (list): list array data.
        Returns:
            tup (tuple): integer tuple data convert from list array. 
        """
        try:
            return tuple(Utilities.toTupleInt(int(i)) for i in a)
        except TypeError:
            return a

    @staticmethod
    def convertStringToTuple(a):
        """
        Convert String to tuple.

        Args:
            a (string): string data.
        Returns:
            tup (tuple): tuple data.
        """

        return literal_eval(a)


    @staticmethod
    def read2D3DPointsFile(dataFileName):
        """
        Read data from specified file contains information of 2D-3D points
        this file is used for calibrating extrinsic
        the (2k+1)-th line (k>=0) is the coordinates of a 3D point and
        the (2k+2)-th line is the coordinates of the corresponding 2D point

        Args:
            dataFileName (string): directory of data file contains information of 2D-3D points. 
        Returns:
            objectPoints (array): object points.
            imagePoints (array): image points.
        """

        objectPoints = []
        imagePoints = []
        if type(dataFileName) is not str:
            print("Error: Input data file name must be string type")
            return 0
        try:
            f = open(dataFileName, 'r')
        except IOError as e:
            print("Error: [Utilities] {0}".format(e.strerror))
            return 0
        i = 0

        for line in f:
            i += 1
            # read each line
            if i%2 == 1:
                objectPoints.append(
                    numpy.array([float(x) for x in line.split()]))
            else:
                imagePoints.append(
                    numpy.array([float(x) for x in line.split()]))
        return objectPoints, imagePoints


    @staticmethod
    def readCSVFile(csvFileName, numOfField = 1, delimiter = ' ', quoteChar = '|'):
        """
         Read all data from a csv file to a list
         each element in the list is a sub-list indicates data of a column field
         all sub-list in the list must have the same length

        Args:
            csvFileName (string): directory of csv file.
            numOfField (int): number of field.
            delimiter (string): delimiter.
            quoteChar (string): quote character.
        Returns:
            data (list): list data read from file.
        """

        data = []
        for i in range(numOfField):
            data.append([])
        spamreader = None
        with open(csvFileName, newline = '') as csvFile:
            spamreader = csv.reader(
                csvFile, delimiter = delimiter, quotechar = quoteChar)
            for row in spamreader:
                for i in range(numOfField):
                    if i >= len(row):
                        data[i].append("None")
                    else:
                        data[i].append(row[i])
        return data

    @staticmethod
    def collectImageNames(csvFileName, *imageFolderPaths):
        """
        Based on the order of input folder paths, collect all images in those folder, label them
        and store all information in a csv file which has each line as a record:
        <label> <image directory>.

        Args:
            csvFileName (string): directory of csv file.
            *imageFolderPaths (string): image folder path.
        Returns:
            None
        """

        outputFile = open(csvFileName, 'w')
        for i in range(len(imageFolderPaths)):
            imagePaths = os.listdir(imageFolderPaths[i])
            for j in range(len(imagePaths)):
                outputFile.write(
                    str(i)+" "+imageFolderPaths[i]+imagePaths[j]+"\n")
        outputFile.close()


    @staticmethod
    def refineImageData(positivePath, negativePath, width, height):
        """
        Crop the center area of all images in positive folder and resize
        divide all images in negative folder to smaller images which
        have equivalent size with positive.

        Args:
            positivePath (string): positive folder path.
            negativePath (string): negative folder path.
            width (int): image width.
            height (int): image height.
        Returns:
            None
        """

        extends = ["*.png", "*.jpg", "*.pgm"]
        positiveImageNames = []
        negativeImageNames = []
        for extend in extends:
            print("Collecting images in "+positivePath+extend)
            positives = glob.glob(positivePath+extend)
            print("Collecting images in "+negativePath+extend)
            negatives = glob.glob(negativePath+extend)
            positiveImageNames += positives
            negativeImageNames += negatives
        for imageName in positiveImageNames:
            image = cv2.imread(imageName)
            if image.shape[0] < height or image.shape[1] < width:
                image = cv2.resize(image, (width, height))
            else:
                dw = image.shape[1]-width
                dh = image.shape[0]-height
                image = image[int(dh/2):int(dh/2)+height, \
                    int(dw/2):int(dw/2)+width]
            cv2.imwrite(imageName, image)
            print("Refined positive image "+imageName)
        for imageName in negativeImageNames:
            image = cv2.imread(imageName)
            if image.shape[0] >= height and image.shape[1] >= width:
                index = 0
                for x in range(0, image.shape[1]-width, width):
                    for y in range(0, image.shape[0]-height, height):
                        subimage = image[y:y+height, x:x+width]
                        cv2.imwrite(
                            imageName[:-4]+"_"+str(index)+imageName[-4:],
                            subimage)
                        index += 1
            print("Refined negative image "+imageName)
            os.remove(imageName)

    @staticmethod
    def showImage(windowName, imageData, waitKey = -1):
        """
        Show image.

        Args:
            windowName (string): window name.
            imageData (numpy array): image to be showed.
            waitKey (int): wait key value.
        Returns:
            None
        """

        if IMAGE_SHOW == 1:
            cv2.imshow(windowName, imageData)
            if (waitKey != -1):
                cv2.waitKey(waitKey)

    @staticmethod
    def getMapFromFile(configFile):
        """
        Get map from file.

        Args:
            configFile (string): directory of configuration file.
        Returns:
            returnedMap (array): map data.
        """

        returnedMap = {}
        with open(configFile, "r") as file:
            for line in file:
                (key, value) = line.split("=")
                returnedMap[key] = value

        return returnedMap

    @staticmethod
    def getValueFromMap(inputMap, key):
        """
        Get value from map.

        Args:
            inputMap (array): input map.
            key (int): keyword to access map data.
        Returns:
            value (float): value take from map. 
        """

        value = None
        try:
            value = inputMap[key]
        except Exception:
            print("Key not exists in map")

        return value

    @staticmethod  
    def calculateLineCoefficients(point1, point2):
        """
        Produces coefs A, B, C of line equation by two points provided

        Args:
            point1 (list of twos floats): first point.
            point2 (list of twos floats): second point.

        Returns:
            A (float): The coefficient A of straight line.
            B (float): The coefficient B of straight line.
            C (float): The coefficient C of straight line.

        """
        A = (point1[1]-point2[1])
        B = (point2[0]-point1[0])
        C = (point1[0]*point2[1]-point2[0]*point1[1])
        return A, B, -C

    @staticmethod  
    def getIntersectionOfTwoLines(line1, line2):
        """
        Finds intersection point (if any) of two lines provided by coefs.

        Args:
            line1 (list of three floats): the coefficients of first line.
            line2 (list of three floats): the coefficients of second line.

        Returns:
            x (float): x value of the intersection point.
            y (float): y value of the intersection point.

            False if no intersection point detected.
        """

        D  = line1[0]*line2[1]-line1[1]*line2[0]
        Dx = line1[2]*line2[1]-line1[1]*line2[2]
        Dy = line1[0]*line2[2]-line1[2]*line2[0]

        if D != 0:
            x = Dx/D
            y = Dy/D
            return x, y
        else:
            return False

    @staticmethod
    def getMostCommon(events, length=5):
        """
        Get most common object in list.

        Args:
            events (list): list of integers.
            length (int): max-length of list

        Returns:
            the most common element.
        """

        mostCommonElement = None
        if events:
            events = events[:length]
            idxs = {k: len(events)-v for v,k in enumerate(events[::-1])}
            c = Counter(events)
            b = list(c.items())
            b.sort(key=lambda x: (-x[1], idxs[x[0]]))
            mostCommonElemet = b[0][0]
        return mostCommonElemet

    @staticmethod    
    def get_sub_path(input_path):
        """"
        This function return sub path from input_path with latest index in input folder.
        
        Args:
            input_path (str): location of folder

        Returns:
            sub_path(str): sub folder with index from input_path

        """
        numbers = []
        sub_path = ""
        for i in sorted(os.listdir(input_path)):
            try:
                number = int(i)
                numbers.append(number)
            except ValueError:
                pass
        if numbers:
            sub_path = os.path.join(input_path, str(numbers[-1]+1))
        else:
            sub_path = os.path.join(input_path, str(0))

        return sub_path

class ByteDumper:
    """
    Dump/load data as binary type.
    """

    @staticmethod
    def dumpVector(vector):
        """
        Dump vector.

        Args:
            vector (vector): vector data.
        Returns:
            byteVector (byte): binary data.
        """

        vector = vector.astype(numpy.float32)
        return vector.tobytes()
    @staticmethod
    def loadVector(byte):
        """
        Load vector.

        Args:
            byte (byte): binary data.
        Returns:
            vector (vector): vector data.
        """

        return numpy.frombuffer(byte, dtype=numpy.float32)

    @staticmethod
    def dumpRectangle(rect):
        """
        Dump rectangle.

        Args:
            rect (array): rectangle as array of int.
        Returns:
            byteRect (byte): rectangle as binary data.
        """

        rect = rect.astype(numpy.int32)
        return rect.tobytes()
    @staticmethod
    def loadRectangle(byte):
        """
        Load rectangle.

        Args:
            byte (byte): rectangle as binary data.
        Returns:
            rect (array): rectangle as array of int.
        """

        return numpy.frombuffer(byte, dtype=numpy.int32)

    @staticmethod
    def dumpCoordinates(coor):
        """
        Dump coordinates.

        Args:
            coor (array): coordinates as array of float.
        Returns:
            byteCoor (byte): coordinates as binary data.
        """

        coor = coor.astype(numpy.float64)
        return coor.tobytes()
    @staticmethod
    def loadCoordinates(byte):
        """
        Load rectangle.

        Args:
            byte (byte): coordinates as binary data.
        Returns:
            coordinates (array): coordinates as array of float.
        """

        return numpy.frombuffer(byte, dtype=numpy.float64)

    @staticmethod
    def dumpMatrix(matrix):
        """
        Dump matrix.

        Args:
            matrix (array): matrix as array.
        Returns:
            byteMatrix (byte): matrix as binary data.
        """

        shape = numpy.array(matrix.shape).astype(numpy.int64)
        sizeShape = numpy.array([len(shape)]).astype(numpy.int64)
        matrix = matrix.astype(numpy.float64)
        return sizeShape.tobytes()+shape.tobytes()+matrix.tobytes()

    @staticmethod
    def loadMatrix(byte):
        """
        Load matrix.

        Args:
            byte (byte): matrix as binary data.
        Returns:
            matrix (array): matrix as array.
        """

        i = numpy.dtype(numpy.int64).itemsize
        sizeShape = numpy.frombuffer(byte[0:i], dtype=numpy.int64)[0]
        shape = numpy.frombuffer(byte[i:i+i*sizeShape], dtype=numpy.int64)
        data = numpy.frombuffer(byte[i+i*sizeShape:], dtype=numpy.float64)
        return data.reshape(shape)

    @staticmethod
    def dumpImage(image):
        """
        Dump image.

        Args:
            image (numpy array): image as numpy array.
        Returns:
            byteImage(byte): image as binary data.
        """

        encoded = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(encoded)

    @staticmethod
    def loadImage(byte):
        """
        Load image.

        Args:
            byte (byte): image as binary data.
        Returns:
            image (numpy array): image as numpy array.
        """

        decodedByte = numpy.fromstring(base64.b64decode(byte), dtype=numpy.uint8)
        return cv2.imdecode(decodedByte, cv2.IMREAD_COLOR)


class TemplateDistance(object):
    """
    Template distance use for distance calculation.

    Attributes:
        __name (string): name of template distance.
    """

    def __init__(self, name):
        """
        Constructor.

        Args:
            name (string): input template distance.
        """

        self.__name = name

    @staticmethod
    def generate(name):
        """
        Generate template distance.

        Args:
            name (string): name of template distance.
        Returns:
            templateDistance (object): return calculate distance object.
        """

        if name == "EuclideanDistance":
            return EuclideanDistance()
        elif name == "ChiSquareDistance":
            return ChiSquareDistance()
        elif name == "HistogramIntersection":
            return HistogramIntersection()
        elif name == "WeightedChiSquare":
            return WeightedChisquare()
        else:
            print("Unknown distance")
            return TemplateDistance()

    def calculate(self, template1, template2):
        """
        Calculate distance.

        Args:
            template1 (vector): template 1.
            template2 (vector): template 2.
        Returns:
            None
        """

        raise NotImplementedError("Every AbstractDistance must implement the calculate method.")


class EuclideanDistance(TemplateDistance):
    """
    Calculate Euclidean distance.
    """

    def __init__(self):
        """
        Constructor.
        """
        TemplateDistance.__init__(self, "EuclideanDistance")

    def calculate(self, template1, template2):
        """
        Calculate Euclidean distance.

        Args:
            template1 (vector): template 1.
            template2 (vector): template 2.
        Returns:
            dist (float): distance.
        """

        template1 = numpy.asarray(template1).flatten()
        template2 = numpy.asarray(template2).flatten()
        dist = Utilities.compareHistograms(template1, template2, cv2.HISTCMP_KL_DIV)
        return dist
        # return numpy.sqrt(numpy.sum(numpy.power((template1-template2), 2)))


class ChiSquareDistance(TemplateDistance):
    """
    Calculate ChiSquare distance.
    """

    def __init__(self):
        """
        Constructor.
        """

        TemplateDistance.__init__(self, "ChiSquareDistance")

    def calculate(self, template1, template2):
        """
        Calculate ChiSquare distance.

        Args:
            template1 (vector): template 1.
            template2 (vector): template 2.
        Returns:
            dist (float): distance.
        """

        template1 = numpy.asarray(template1).flatten()
        template2 = numpy.asarray(template2).flatten()
        bin_dists = (template1-template2)**2 \
            /(template1+template2+numpy.finfo('float').eps)
        return numpy.sum(bin_dists)/100
        # dist = Utilities.compareHistograms(template1, template2, cv2.HISTCMP_CHISQR)
        # return dist/100


class HistogramIntersection(TemplateDistance):
    """
    Calculate histogram intersection distance.
    """

    def __init__(self):
        """
        Constructor.
        """

        TemplateDistance.__init__(self, "HistogramIntersection")

    def calculate(self, template1, template2):
        """
        Calculate histogram intersection.

        Args:
            template1 (vector): template 1.
            template2 (vector): template 2.
        Returns:
            dist (float): distance.
        """

        template1 = numpy.asarray(template1).flatten()
        template2 = numpy.asarray(template2).flatten()
        return numpy.sum(numpy.minimum(template1, template2))


class WeightedChisquare(TemplateDistance):
    #Special class for LBP matching
    """
    Calculate Weighted Chisquare distance.
    """

    RATING_TEMPLATE = [
        2,4,4,1,4,4,2,\
        1,1,1,0,1,1,1,\
        0,1,1,1,1,1,0,\
        0,1,1,1,1,1,0,\
        0,1,1,2,1,1,0,\
        0,1,1,1,1,1,0,\
        0,0,1,1,1,0,0]

    def __init__(self):
        """
        Constructor.
        """

        TemplateDistance.__init__(self, "WeightedChiSquare")

    def calculate(self, template1, template2):
        """
        Calculate Weighted Chisquare distance.

        Args:
            template1 (vector): template 1.
            template2 (vector): template 2.
        Returns:
            dist (float): distance.
        """
        if (len(template1) != len(template2)) or (len(template1) % 59 != 0):
            print("len(template1) = ", len(template1))
            print("len(template2) = ", len(template2))
            return -1

        stepSize = int(len(template1) / 59)
        dist = 0
        for i in range(0, stepSize):
            # print("I = ", i)
            smallTemp1 = template1[i : i + 59]
            smallTemp2 = template2[i : i + 59]
            # bin_dists = (smallTemp1 - smallTemp2) ** 2 / (smallTemp1 + smallTemp2 + numpy.finfo('float').eps)
            # tempDist = WeightedChisquare.RATING_TEMPLATE[i] * numpy.sum(bin_dists)
            tempDist = WeightedChisquare.RATING_TEMPLATE[i] * cv2.compareHist(smallTemp1, smallTemp2, cv2.HISTCMP_CHISQR)
            dist += tempDist

        # print("Dist = ", dist)
        return dist / 100


class TemplateObjectDetect(object):
    """
    Template object detector.
    """

    def __init__(self, name):
        """
        Constructor.

        Args:
            name (string):
        """

        self.name = name

    def detect(self, image):
        """
        Detect objects in given image.

        Args:
            image (numpy array): image contains objects.
        Returns:
            None
        """
        raise NotImplementedError("Abstract detection")

    def normalize(self, image, position):
        """

        Args:
            image (numpy array): image contains objects.
            position (array): object position in image.
        Returns:
            None
        """
        raise NotImplementedError("Abstract normalization")


class KNN:
    """
    K-nearest neighbors classifier.

    Attributes:
        mK (int): number of nearest number, is a constant defined value from user.
        mMetric (string): metric to calculate distance.
        mFeature (list): list of training features.
        mLabel (list): list of training labels.
    """

    def __init__(self, k, metrics = 'EuclideanDistance'):
        """
        Constructor.

        Args:
            k (int): user input K number.
            metrics (string): chosen metric for KNN classfier.
        """

        self.mK = k
        self.mMetric = TemplateDistance.generate(metrics)
        self.mFeature = []
        self.mLabel = []

    def setData(self, featureMatrix, featureLabel):
        """
        Set data for KNN classifier

        Args:
            featureMatrix (list): list of training features.
            featureLabel (list): list of training labels.
        Returns:
            None.
        """

        self.mFeature = []
        self.mLabel = []
        for i in range(len(featureMatrix)):
            self.mFeature.append(featureMatrix[i])
            self.mLabel.append(featureLabel[i])
        # for i in self.mLabel:
        #     print("Label = ", i)

    def predict(self, featureVector):
        """
        Predict object label.

        Args:
            featureVector (vector): feature vector of object to predict.
        Returns:
            label (int): predicted label of object.
            maxConf (float): relate distance of predicted object label.
        """

        # Pick k nearest
        distances = []

        for i in range(len(self.mFeature)):
            dist = self.mMetric.calculate(self.mFeature[i], featureVector)
            distances.append((dist, self.mLabel[i])) # distance = [[dist, label]....]

        if len(distances) == 1:
            return distances[0][0], distances[0][1]
        distances.sort(key=operator.itemgetter(0))

        neighbors = []
        if self.mK > len(distances):
            for i in range(len(distances)):
                neighbors.append(distances[i])
        else:
            for i in range(self.mK):
                neighbors.append(distances[i])

        # Majority vote
        votes = {}
        for i in range(len(neighbors)):
            singleLabel = neighbors[i][1]
            if singleLabel in votes:
                votes[singleLabel] += 1
            else:
                votes[singleLabel] = 1
        sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)

        # Get the maximize confidence for that vote
        label = sortedVotes[0][0]
        confList = []
        for i in range(len(neighbors)):
            if neighbors[i][1] == label:
                confList.append(neighbors[i][0])
        maxConf = min(confList)

        return maxConf, label
        
    def predictNumObjects(self, featureVector, numObj):
        distances = []
        for i in range(len(self.mFeature)):
            dist = self.mMetric.calculate(self.mFeature[i], featureVector)
            distances.append((dist, self.mLabel[i])) # distance = [[dist, label]....]

        if len(distances) == 1:
            return(distances[0][0], distances[0][1]) 
        distances.sort(key=operator.itemgetter(0))

        resultList = []
        while numObj > 0:
            minK = min(len(distances), self.mK)
            self.mK = minK
            neighbors = distances[0:self.mK]

            # Majority vote
            votes = {}
            for neighbor in neighbors:
                singleLabel = neighbor[1]
                if singleLabel in votes:
                    votes[singleLabel] += 1
                else:
                    votes[singleLabel] = 1
            votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)

            # Handle in case vote have more than one biggest value
            label = -1
            if len(votes) == 1 or votes[0][1] != votes[1][1]:
                label = votes[0][0]
            else:
                maxVote = votes[0][1]
                votedNeighbors = []
                for vote in votes:
                    if vote[1] == maxVote:
                        votedNeighbors += [neighbor for neighbor in neighbors if neighbor[1] == vote[0]]
                votedNeighbors.sort(key=operator.itemgetter(0))
                label = votedNeighbors[0][1]
            
            # Get the maximize confidence for that vote
            confList = []
            confList = [neighbor[0] for neighbor in neighbors if neighbor[1] == label]
            confidence = min(confList)
            # Remove picked item
            distances = [dist for dist in distances if dist[1] != label]
            prediction = (label, confidence)
            resultList.append(prediction)
            numObj -=1

        return resultList

    def predictList(self, featureVector, mObjects, threshold):
        """
        Pick m common objects in k nearest neighbors.

        Args:
            featureVector (list of: obj): list of feature vector.
            mObjects (int): m common objects.
            threshold (float): threshold of confidence

        Returns:
            dists (list of: obj):  list of m most common object ids with \
            their distances
        """
        distances = []
        mObjects = min(mObjects, len(self.mFeature))

        # calculate distance for feature matrix and input feature
        for i in range(len(self.mFeature)):
            dist = self.mMetric.calculate(self.mFeature[i], featureVector)
            if (dist <= threshold):
                distances.append((dist, self.mLabel[i]))

        # sort distances by distance
        distances.sort(key=operator.itemgetter(0))

        # keep kNN of distances
        if (self.mK < len(distances)):
            del distances[self.mK:len(distances)]

        # get m most common objects in list of distance
        objIDs = (dist[1] for dist in distances)
        kNN = Counter(objIDs)
        mCommonObjIds = kNN.most_common(mObjects)

        # unique distances by object id with the best distance
        seen = set()
        uniqueDist = [item for item in distances if item[1] not in seen and \
        not seen.add(item[1])]

        # find m most common object ids with its distance
        dists = list((obj[0], dist[0]) for obj in mCommonObjIds for dist in \
            uniqueDist if obj[0] == dist[1])

        # get the number of object ids or the best distance
        dists.sort(key=operator.itemgetter(1))

        return dists