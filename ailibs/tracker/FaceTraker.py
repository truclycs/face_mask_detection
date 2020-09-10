# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

from ailibs.__init__ import timeit

UNKNOWN = "Unknown"
ACCEPT = "ACCEPT"
ALERT = "ALERT"


class FaceTracker():
    """
    This is implementation for tracking face.

    """

    def __init__(self, **kwargs):
        """
        Constructor.
        Args:

        """
        self.log = kwargs.get('log', False)
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = kwargs.get('maxDisappeared', 20)

    def register(self, centroid):
        """ add new ID by centroid

        Args:
            centroid (OrderedDict()): an order dict() of centroid point
        """
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """ remove ID by objectID

        Args:
            objectID (int): ID of object
        """
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    @timeit
    def update(self, rects):
        """ update dictionary of object by rects

        Args:
            rects (dlib.rectangles): dlib rectangles from face detector

        Returns:
            OrderedDict(): dict of ID and its centroid
        """
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        # for (i, d) in enumerate(rects):
        # use the bounding box coordinates to derive the centroid
        for (i, rect) in enumerate(rects):
            startX = rect.left()
            endX = rect.right()
            startY = rect.bottom()
            endY = rect.top()
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                # self.usernames[objectID] = ID[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

    def track_face(self, ID, trackcount, dets, user_list, name_list, flag, frame):
        # update centroid tracker using the computed set of bounding
        # box rectangles
        ACCEPT_COUNT = 2
        ALERT_COUNT = 10
        for (faceID, centroid) in self.objects.items():
            face_utils = zip(dets, user_list)
            for i, face_util in enumerate(face_utils):
                # loop over the tracked objects
                # draw both the ID of the faces and the centroid of the
                # object on the output frame
                # centroid = centroid[0]
                d, NAME = face_util
                name = NAME['name']
                [startX, endX, startY, endY] = [
                    d.left(), d.right(), d.bottom(), d.top()]
                [cX, cY] = [int((startX + endX) / 2.0),
                            int((startY + endY) / 2.0)]

                if cX - centroid[0] == 0 and cY - centroid[1] == 0:
                    if faceID not in ID.keys():
                        ID[faceID] = UNKNOWN
                        trackcount[faceID] = 0
                    if flag:
                        if (ID[faceID] != name):
                            trackcount[faceID] -= 1
                        if (ID[faceID] == name):
                            trackcount[faceID] += 1
                        if (trackcount[faceID] < 0):
                            ID[faceID] = name
                            trackcount[faceID] = 0
                        if ID[faceID] not in name_list:
                            continue
                        if (trackcount[faceID] > ALERT_COUNT) and (ID[faceID] == UNKNOWN) and (ALERT not in name_list[ID[faceID]]['name']):
                            name_list[ID[faceID]]['name'] = name_list[ID[faceID]
                                                                      ]['name'] + " " + ALERT
                        if (trackcount[faceID] > ACCEPT_COUNT) and (ID[faceID] != UNKNOWN) and (ACCEPT not in name_list[ID[faceID]]['name']):
                            name_list[ID[faceID]]['name'] = name_list[ID[faceID]
                                                                      ]['name'] + " " + ACCEPT
        flag = False
        return ID, trackcount, flag
