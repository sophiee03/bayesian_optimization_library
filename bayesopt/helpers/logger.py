import logging
import sys

def setup_console_logger(name: str = 'BO') -> logging.Logger:
    """creates the logger to show execution information
    
    Args:
        name (str): name of the logger

    Returns:
        Logger: logger created
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger
