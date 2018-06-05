# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Implements functions for predicting the future re-tweets

References
----------
.. *Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""
import math


def prediction_lr_n(parameter_estimated, no_events, total_follower_t, follower_orig):
    """
    predict the total number of possible re-tweet at the time t
    :param parameter_estimated: parameter for linear regression model
    :param no_events: array containing the total number of re-tweet intil the observation time
    :param total_follower_t: array containing the total number of follower until the observation time
    :param follower_orig: array containing the original number of follower of the original tweet
    :return: the biased estimator
    """

    est_rf = math.exp(parameter_estimated[0][0][0] + (parameter_estimated[0][1][0] * no_events) +
                      (parameter_estimated[0][2][0] * total_follower_t) +
                      (parameter_estimated[0][3][0] * follower_orig) + (parameter_estimated[1][0] / 2))
    return est_rf


def prediction_lr_n_cross_validation(parameter_estimated, no_events, total_follower_t, follower_orig):
    """
    predict the total number of possible re-tweet at the time t
    :param parameter_estimated: parameter for linear regression model
    :param no_events: array containing the total number of re-tweet intil the observation time
    :param total_follower_t: array containing the total number of follower until the observation time
    :param follower_orig: array containing the original number of follower of the original tweet
    :return: the biased estimator
    """

    est_rf = [math.exp(parameter_estimated[0][j][0] + (parameter_estimated[0][j][1] * no_events)
                       + (parameter_estimated[0][j][2] * total_follower_t)
                       + (parameter_estimated[0][j][3] * follower_orig) + (
                               parameter_estimated[1][j] / 2))
              for j in range(len(parameter_estimated[0]))]
    return est_rf
