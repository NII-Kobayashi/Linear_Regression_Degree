# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Implements functions for estimating the parameters used in linear regression model

References
----------
.. *Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""
import numpy as np
from scipy import linalg


def parameter_estimation_lr_n(follower_orig, total_follower_t, no_events, event_pred):
    """
    calculate the parameters value for the linear regression model
    :param follower_orig: array containing the original number of follower of the original tweet
    :param total_follower_t: array containing the total number of follower until the observation time
    :param no_events: array containing the total number of re-tweet intil the observation time
    :param event_pred: array containing the actual value of total number of re-tweet at the prediction time
    :return: the linear regression model parameters (theta and variance)
    """

    a = np.matrix([
        [len(follower_orig), sum(no_events), sum(total_follower_t), sum(follower_orig)],
        [sum(no_events), sum(no_events * no_events), sum(total_follower_t * no_events),
         sum(follower_orig * no_events)],
        [sum(total_follower_t), sum(no_events * total_follower_t), sum(total_follower_t * total_follower_t),
         sum(total_follower_t * follower_orig)],
        [sum(follower_orig), sum(no_events * follower_orig), sum(total_follower_t * follower_orig),
         sum(follower_orig * follower_orig)]
    ])

    y1 = sum(event_pred)
    y2 = sum([(no_events[i] * event_pred[i]) for i in range(len(event_pred))])
    y3 = sum([(total_follower_t[i] * event_pred[i]) for i in range(len(event_pred))])
    y4 = sum([(follower_orig[i] * event_pred[i]) for i in range(len(event_pred))])

    if isinstance(y1, np.float64):
        theta = linalg.solve(a, (np.matrix([y1, y2, y3, y4])).transpose())
        sigma_val = [(event_pred[i] - theta[0] - (theta[1] * no_events[i]) - (theta[2] * total_follower_t[i]) -
                      (theta[3] * follower_orig[i])) ** 2 for i in range(len(event_pred))]
        sigma_sq = sum(sigma_val) / len(event_pred)

        return theta, sigma_sq

    else:
        theta = [linalg.solve(a, (np.matrix([y1, y2, y3, y4])).transpose())]
        sigma_sq_list = []
        for j in range(len(theta)):
            val = [(event_pred[i][j] - theta[j][0] - (theta[j][1] * no_events[i]) -
                    (theta[j][2] * total_follower_t[i]) - (theta[j][3] * follower_orig[i])) ** 2
                   for i in range(len(no_events))]
            sigma_sq = sum(val) / len(no_events)
            sigma_sq_list.append(sigma_sq)

        return theta, sigma_sq_list

