from time import time
from datetime import datetime


def get_current_date_and_time():
    current_time_in_seconds = time()
    current_time_in_datetime = datetime.fromtimestamp(current_time_in_seconds)

    return current_time_in_datetime.strftime('%Y_%m_%d_%H_%M_%S')
