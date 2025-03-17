import logging

"""
Logger takes filename (by convention "/logger_[function_name]_t.txt" where t is time initialised), and program_name
"""
def set_up_logger(filename, program_name):
    # REF: https://stackoverflow.com/questions/55169364/python-how-to-write-error-in-the-console-in-txt-file
    logger = logging.getLogger(program_name)
    logger.setLevel(logging.INFO) # you can set this to be DEBUG, INFO, ERROR
    # Assign a file-handler to that instance
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO) # again, you can set this differently
    # Format your logs (optional)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter) # This will set the format to the file handler
    # Add the handler to your logging instance
    logger.addHandler(fh)
    return logger