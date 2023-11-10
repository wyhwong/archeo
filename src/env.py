import os

MAX_WORKER = int(os.getenv("MAX_WORKER", "20"))
LOGLEVEL = int(os.getenv("LOGLEVEL", "20"))
LOGFILEPATH = os.getenv("LOGFILEPATH", "")
