# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Functions for estimating the parameters (alpha_t, sigma^2_t, beta^1_t, beta^2_t, beta^3_t) of the linear regression with
degree (LR-N)

References
----------
.. *Kobayashi and Lambiotte, ICWSM, pp. 191-200, 2016; Zhao et al., KDD, pp. 1513-1522, 2015*.
"""

import numpy as np
from scipy import linalg


def parameter_estimation_lr_n(follower_orig, total_follower_t, no_events, event_pred):
    """
    Fit the parameters of the linear regression with degree
    :param follower_orig: array, containing the number of followers of the original tweeting person for all tweet (d_0).
    :param total_follower_t: array, containing the cumulative number of followers at an observation time for all tweet (D(T)).
    :param no_events: array, containing the total number of retweets at an observation time (R(T) ) for all tweet.
    :param event_pred: array, containing the total number of retweets at a prediction time (R(t) ) for all tweet.
    :return: parameters of the linear regression with degree.
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

    # y1 = sum(event_pred)
    # y2 = sum([(no_events[i] * event_pred[i]) for i in range(len(event_pred))])
    # y3 = sum([(total_follower_t[i] * event_pred[i]) for i in range(len(event_pred))])
    # y4 = sum([(follower_orig[i] * event_pred[i]) for i in range(len(event_pred))])

    # same as above with numpy to avoid loops
    y1 = np.sum(event_pred, axis=0)
    y2 = np.sum((event_pred.T * no_events).T, axis=0)
    y3 = np.sum((event_pred.T * total_follower_t).T, axis=0)
    y4 = np.sum((event_pred.T * follower_orig.T).T, axis=0)

    if isinstance(y1, np.float64):
        theta = linalg.solve(a, [y1, y2, y3, y4])
        sigma_val = [(event_pred[i] - theta[0] - (theta[1] * no_events[i]) - (theta[2] * total_follower_t[i]) -
                      (theta[3] * follower_orig[i])) ** 2 for i in range(len(event_pred))]
        sigma_sq = sum(sigma_val) / len(event_pred)

        return [theta], [sigma_sq]

    else:
        theta = [linalg.solve(a, [y1[i], y2[i], y3[i], y4[i]]) for i in range(len(sum(event_pred)))]
        sigma_sq_list = []
        for j in range(len(theta)):
            val = [(event_pred[i][j] - theta[j][0] - (theta[j][1] * no_events[i]) -
                    (theta[j][2] * total_follower_t[i]) - (theta[j][3] * follower_orig[i])) ** 2
                   for i in range(len(no_events))]
            sigma_sq = sum(val) / len(no_events)
            sigma_sq_list.append(sigma_sq)

        return theta, sigma_sq_list
