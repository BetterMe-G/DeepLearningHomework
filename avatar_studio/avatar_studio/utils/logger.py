import logging
import sys

_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATE = "%H:%M:%S"


def get_logger(name: str = "avatar") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
