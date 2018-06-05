# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Full example to checks the accuracy of the model.
Please replace file paths according to your local directory structure.

References
----------
.. *Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""

from cross_validation import *
from function import*
import glob as gb


def main(obs_time, file_list):
    """
    print the mean, median and correlation error at the observation
    :param obs_time: observation time
    :param file_list: data files
    """
    for q in range(0, 6):
        dt = (2 ** q)
        if q == 4:
            dt = 24
        if q == 5:
            dt = 162
        max_value = int((168 - obs_time) / dt)  # number of bins
        window_size = dt  # dt in second
        event_list = [no_of_events_followers_in_window(file_list[i], obs_time, window_size, max_value, 3600) for i in
                      range(len(file_list))]
        event_list = list(filter(None.__ne__, event_list))  # checking for none value
        result_lr = cross_validation_error(5, event_list, max_value)
        print("Time:", dt, result_lr)


file_path = "Data/RT*.txt"
file_list_all = sorted(gb.glob(file_path), key=numerical_sort)
main(6, file_list_all)

