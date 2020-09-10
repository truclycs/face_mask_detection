import time
from datetime import datetime

def timeit(func):
    """
    Log executing time if log is enable.
    """
    def timed(self, *args, **kwargs):
        ts = time.time()
        result = func(self, *args, **kwargs)
        te = time.time()
        if self.log:
            print("[{}] {}() is {} ms.".format(datetime.now(), func.__qualname__, (int((te - ts) * 1000))))
        return result
    return timed
