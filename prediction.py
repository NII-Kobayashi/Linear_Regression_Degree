# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Functions for predicting the number of retweets in future by using linear regression with degree (LR-N)

References
----------
.. *Kobayashi and Lambiotte, ICWSM, pp. 191-200, 2016; Zhao et al., KDD, pp. 1513-1522, 2015*.
"""
import math


def prediction_lr_n(parameter_estimated, no_events, total_follower_t, follower_orig):
    """
    Predicting the total number of retweets
    :param parameter_estimated: the parameters of linear regression with degree
    :param no_events: array, containing the total number of retweet until an observation time for all tweet
    :param total_follower_t: array, containing the cumulative number of followers until the observation time for all
    tweet
    :param follower_orig: array containing the number of followers of the original tweeting person for all tweet
    :return: Predicted number of retweets
    """
    pe, sig = parameter_estimated

    est = [math.exp(alpha + (beta_r * no_events) + (beta_n * total_follower_t) + (beta_0 * follower_orig) + (
            sig / 2)) for (alpha, beta_r, beta_n, beta_0), sig in zip(pe, sig)]
    return est
