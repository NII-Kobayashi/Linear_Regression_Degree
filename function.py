# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Functions for calculating the total number of retweets and the number of followers.

Functions for predicting the number of retweets in a future period based on a single window (no_of_events_followers) or
multiple windows (no_of_events_followers_in_window).

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
    numeric sort the filename path in the directory.
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def no_of_events_followers(event_file, t_observation, t_prediction, time_factor=1):
    """
    calculate the number of retweets at an observation time (=t_observation) and the prediction time (=t_prediction)
    based on a single window
    :param tweet_file: data file for retweets
    :param t_observation: an observation time
    :param t_prediction: the prediction time
    :param time_factor: the factor to convert the time unit in seconds
    :return: tuple, containing the logarithm of the followers of an original tweeted person (log(d_0)), the logarithm
    of the cumulative number of followers (log(D(T))), the logarithm of the number of retweets at an observation
    time (log(R(T))), the number of retweets at the prediction times (R(t) ), and its logarithm (log(R(t)))
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
    calculate the number of retweets at an observation time (=t_observation) and at the  prediction times (=t_prediction)
    based on multiple windows.
    :param tweet_file: data file for retweets
    :param t_observation: observation time
    :param t_prediction: prediction times
    :param time_factor: factor to convert the time unit in seconds
    :param win_size: window size for prediction
    :param max_itr: the number of windows used in prediction
    :return: list, containing the logarithm of the followers of an original tweeted person (log(d_0)), the logarithm
    of the cumulative number of followers (log(D(T))), the logarithm of the number of retweets at an observation
    time (log(R(T))), the number of retweets until the prediction times (R(t)), and its logarithm (log(R(t)))
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
