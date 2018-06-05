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
    pe, sig = parameter_estimated

    est = [math.exp(alpha + (beta_r * no_events) + (beta_n * total_follower_t)  +( beta_0* follower_orig) + (
            sig / 2)) for (alpha, beta_r, beta_n, beta_0), sig in zip(pe, sig)]
    return est
