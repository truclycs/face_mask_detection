import os
import logging

from utils.Utilities import Utilities

configFile = os.path.abspath(
        os.path.join(os.path.abspath(os.path.realpath(__file__)), "..", "..", "utils", "log.conf"))

log = logging.getLogger("fci_streaming")
settingmaps = Utilities.getMapFromFile(configFile)

logTemplate = settingmaps["template"]
logFormatter = logging.Formatter(logTemplate)

outFile = settingmaps["outfile"].strip()
outPath = os.path.abspath(
        os.path.join(os.path.abspath(os.path.realpath(__file__)), "..", "..", "utils", outFile))

fileHandler = logging.FileHandler(outPath)
fileHandler.setFormatter(logFormatter)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

level = Utilities.getValueFromMap(settingmaps, "level")

if level == "NOTSET":
    log.setLevel(logging.NOTSET)
elif level == "DEBUG":
    log.setLevel(logging.DEBUG)
elif level == "INFO":
    log.setLevel(logging.INFO)
elif level == "WARNING":
    log.setLevel(logging.WARNING)
elif level == "ERROR":
    log.setLevel(logging.ERROR)
elif level == "CRITICAL":
    log.setLevel(logging.CRITICAL)
else:
    log.setLevel(logging.INFO)

log.addHandler(consoleHandler)
log.addHandler(fileHandler)