import logging
import sys

def setup_console_logger(name: str = 'BO', level: int = logging.INFO) -> logging.Logger:
    '''creates the logger to show execution information'''
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger
