import logging

def get_logger(name:str) -> logging.Logger:
    #Create a logger and sets its level - Shows INFO, WARNING etc but not debug messages
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    #Sets up handlers if not done already
    #Prevents duplicate lines on multiple configurations
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

#Controls the format of the loggers outputs
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger