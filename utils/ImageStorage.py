import os
import cv2


# from pymemcache.client.base import Client
import logging as log


class ImageStorage:
    """
    Handle save/load image image into server storage location.

    Attributes:
        __source (string): directory of server storage location .
        __extend (string): format of save image.
    """


    MAXIMUM = 9223372036854775807
    MAXIMUM_LENGTH = 19
    FRAGMENT_LENGTH = 3

    def __init__(self, source, extend=".jpg"):
        """
        Constructor.

        Args:
            source (): initial value for __source.
            extend (): initial value for __extend.
        """
        self.__source = source
        self.__extend = extend    

    def write(self, image, image_id):
        """
        Save image image into storage location.

        Args:
            image (numpy array): image of image.
            image_id (int): Id of image.
        Returns:
            None
        """

        max_length = ImageStorage.MAXIMUM_LENGTH
        frag_length = ImageStorage.FRAGMENT_LENGTH
        image_name = str(image_id).zfill(max_length)
        directory = self.__source
        for i in range(0, max_length, frag_length):
            if max_length-i < frag_length:
                break
            directory = os.path.join(directory, image_name[i:i+frag_length])
        image_name = os.path.join(directory, image_name)+self.__extend
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as exc:
                log.error("Cannot make directory: {0}".format(exc))
        try:
            cv2.imwrite(image_name, image)
        except Exception as exc:
            log.error("Cannot write image image: {0}".format(exc))

    def read(self, image_id):
        """
        Load image image into storage location.

        Args:
            image_id (int): Id of image .

        Returns:
            image (numpy array): image.
        """

        max_length = ImageStorage.MAXIMUM_LENGTH
        frag_length = ImageStorage.FRAGMENT_LENGTH
        image_name = str(image_id).zfill(max_length)
        directory = self.__source
        for i in range(0, max_length, frag_length):
            if max_length-i < frag_length:
                break
            directory = os.path.join(directory, image_name[i:i+frag_length])
        image_name = os.path.join(directory, image_name)+self.__extend
        return cv2.imread(image_name)

    def delete(self, image_id):
        """
        Delete image image from storage location.

        Args:
            image_id (int): Id of image.
        Returns:
            None
        """

        max_length = ImageStorage.MAXIMUM_LENGTH
        frag_length = ImageStorage.FRAGMENT_LENGTH
        image_name = str(image_id).zfill(max_length)
        directory = self.__source
        for i in range(0, max_length, frag_length):
            if max_length-i < frag_length:
                break
            directory = os.path.join(directory, image_name[i:i+frag_length])
        image_name = os.path.join(directory, image_name)+self.__extend
        # log.info("image path: %s" % image_name)
        if not os.path.exists(directory):
            raise Exception('The storage location: %s does not exist!' % directory)
        try:
            os.remove(image_name)
        except Exception as exc:
            log.error("Cannot delete image: {0}".format(exc))

# class imageDataStorage:
#     """
#     Handle save/load current streaming image of camera, using pymemcache.

#     Attributes:
#         __client (object): pymemcache object.
#     """


#     def __init__(self, host="localhost", port=11211):
#         """
#         Constructor.

#         Args:
#             host (string): server IP.
#             port (int): server Port.
#         """

#         self.__client = Client((host ,port))

#     def storeData(self, key, data):
#         """
#         Save current image streaming data.

#         Args:
#             key (int): keyword to save current image data, as camera Id.
#             data (numpy array): image of current streaming image.
#         Returns:
#             True if success.
#             False if fail.
#         """

#         try:
#             # log.info("key %s data %s" %(key, data))
#             self.__client.set(key, data)
#         except Exception as exc:
#             log.error("Cannot store data: {0}".format(exc))
#             return False
#         return True

#     def loadData(self, key):
#         """
#         Load current image streaming data.
#         Args:
#             key (int): keyword to load current image data, as camera Id.

#         Returns:
#             data (numpy array): image of current streaming image.
#         """

#         data = None
#         try:
#             data = self.__client.get(key)
#         except Exception as exc:
#             log.error("Cannot load data: {0}".format(exc))
#         return data

#     def deleteData(self, key):
#         """
#         Delete current image streaming data. 
#         Args:
#             key (int): keyword to delete image data, as camera Id.

#         Returns:
#             True if success.
#             False if fail.

#         """

#         try:
#             result = self.__client.delete(key, noreply=False)
#         except Exception as exc:
#             log.error("Cannot delete data: {0}".format(exc))
#             return False
#         return result
                
