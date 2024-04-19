import logging

import archeo.env


LOGFMT = "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%SZ"
logging.basicConfig(format=LOGFMT, datefmt=DATEFMT, level=archeo.env.LOGLEVEL)


def get_logger(logger_name: str, log_filepath=archeo.env.LOGFILE_PATH) -> logging.Logger:
    """
    Get logger

    Args:
    -----
        logger_name (str):
            Logger name

        log_filepath (str):
            Log filepath

    Returns:
    -----
        logger (logging.Logger):
            Logger
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(archeo.env.LOGLEVEL)

    if log_filepath:
        handler = logging.FileHandler(filename=log_filepath)
        formatter = logging.Formatter(fmt=LOGFMT, datefmt=DATEFMT)
        handler.setFormatter(fmt=formatter)
        logger.addHandler(handler)

    return logger
