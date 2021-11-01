import cv2
from streaming import PYTHON_PATH

ACCEPT_COUNT = 2
ALERT_COUNT = 5
ALERT = "ALERT"
ACCEPT = "ACCEPT"
UNKNOWN = "Unknown"


class Alert():
    """
    This is implementation for alert message.
    """

    def __init__(self, **kwargs):
        """
        Constructor.
        """
        # alert images storage
        self.alert_path = kwargs.get('images')

    def alert(self, trackcount, ID, NAME_LIST, flag, faceID, frame):
        """[summary]

        Args:
            ID (dict): tracker ID
            trackcount (dict):  verified score of face
            NAME_LIST (dict): check-in users
            flag (bool): check sending image by mail
            faceID (int): ID of tracked face
            frame (cv2): frame

        Returns:
            flag (bool): check sending image by mail
            NAME_LIST (dict): check-in users
        """
        if (trackcount[faceID] > ALERT_COUNT) and (ID[faceID] == UNKNOWN) and (ALERT not in NAME_LIST[ID[faceID]]['name']):
            NAME_LIST[ID[faceID]]['name'] = NAME_LIST[ID[faceID]
                                                      ]['name'] + " " + ALERT
            if flag[faceID]:
                cv2.imwrite(
                    f"{self.alert_path}/{faceID}.jpg", frame)
                flag[faceID] = False
        if (trackcount[faceID] > ACCEPT_COUNT) and (ID[faceID] != UNKNOWN) and (ACCEPT not in NAME_LIST[ID[faceID]]['name']):
            NAME_LIST[ID[faceID]]['name'] = NAME_LIST[ID[faceID]
                                                      ]['name'] + " " + ACCEPT
        return flag, NAME_LIST
