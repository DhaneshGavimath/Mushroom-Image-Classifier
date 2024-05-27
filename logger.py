import logging
import datetime

logger = logging.getLogger()

# file handler
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%d-%m-%Y")
file_name = "logs_{}.log".format(formatted_date)
logger_path = "./logs/{}".format(file_name)
handler = logging.FileHandler(filename=logger_path)
stream_handler = logging.StreamHandler() # to write to console

# formatting message
formatter = logging.Formatter(fmt="%(asctime)s %(filename)s %(levelname)s : %(message)s")
handler.setFormatter(formatter)

# Adding stream and file handler to logger
logger.addHandler(handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
