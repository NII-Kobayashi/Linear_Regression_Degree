# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Implements functions for calculating the number of events and number of followers within the observation time and the
prediction time.

Provides different function for calculating the number of events, with single and multiple prediction time with the
window concept

References
----------
.. *Kobayashi and Lambiotte, ICWSM, pp. 191-200, 2016; Zhao et al., KDD, pp. 1513-1522, 2015*.
"""

import re
import warnings
import math

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    """
    numerically sort the filename path in the directory.
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def no_of_events_followers(event_file, t_observation, t_prediction, time_factor=1):
    """
    calculate the number of retweets at the observation time (=t_observation) and at the final time of prediction
    (=t_prediction)
    :param tweet_file: data file
    :param t_observation: observation time
    :param t_prediction: final time of prediction
    :param time_factor: factor to convert the time unit in seconds
    :return: tuple, containing the logarithms of, follower of an original tweet, number of followers and events until the
    observation time, number of events until the prediction times
    """
    event_no_t_obs = 0
    follower_t = 0  # will store number of follower before observation time
    original_follower = 0
    event_t_pred = 0
    with open(event_file, "r") as in_file:
        first = next(in_file)  # to remove the first line in the tweet file
        for num, line in enumerate(in_file, 1):
            values = line.split(" ")
            if num == 1:
                original_follower = int(values[1])
            if float(values[0]) <= (t_observation * time_factor):
                follower_t = follower_t + int(values[1])
                event_no_t_obs = num - 1  # to remove the original tweet
            if float(values[0]) <= (t_prediction * time_factor):
                event_t_pred = num - 1  # to remove the original tweet

    if event_no_t_obs == 0:
        warnings.warn("No event have occurred till the observation time. The file WILL BE IGNORED")
        print("Ignored File Name:", event_file)
    else:
        return math.log(original_follower), math.log(follower_t), math.log(event_no_t_obs), math.log(event_t_pred), \
               event_t_pred


def no_of_events_followers_in_window(event_file, t_observation, win_size, max_itr, time_factor=1):
    """
    calculate the number of retweets at the observation time (=t_observation) and at final times of prediction
    (=t_prediction)
    :param tweet_file: data file
    :param t_observation: observation time
    :param t_prediction: final time of prediction
    :param time_factor: factor to convert the time unit in seconds
    :param win_size: window size for prediction
    :param max_itr: the number of windows used in prediction
    :return: list containing  the logarithms of, follower of an original tweet, number of followers and events until the
    observation time, number of events with their logarithms until the and prediction times
    """
    event_no_t_obs = 0
    follower_t = 0
    original_follower = 0
    event_t_pred = 0
    event_t_pred_list = []
    event_t_pred_list_log = []
    time = t_observation * time_factor  # converting in seconds
    win_size_sec = win_size * time_factor
    t_f_list = [time + (i * win_size_sec) for i in range(1, max_itr + 1)]
    for i in range(len(t_f_list)):
        with open(event_file, "r") as in_file:
            first = next(in_file)  # to remove the first line containing total no of re-tweet and no of follower
            for num, line in enumerate(in_file, 1):
                values = line.split(" ")
                if num == 1:
                    original_follower = int(values[1])
                if i == 0:
                    if float(values[0]) <= time:
                        event_no_t_obs = num - 1
                        follower_t = follower_t + int(values[1])
                if float(values[0]) <= t_f_list[i]:
                    event_t_pred = num - 1

            if event_no_t_obs == 0:  # ignoring the file if there is no event happened during the observation time
                warnings.warn("No event have occurred till the observation time. The file WILL BE IGNORED")
                print("Ignored File Name:", event_file)
                break
            else:
                event_t_pred_list_log.append(math.log(event_t_pred))
                event_t_pred_list.append(event_t_pred)

    if event_no_t_obs == 0:  # ignoring the file if there is no event happened during till the observation time
        pass
    else:
        return math.log(original_follower), math.log(follower_t), math.log(event_no_t_obs),\
               event_t_pred_list_log, event_t_pred_list