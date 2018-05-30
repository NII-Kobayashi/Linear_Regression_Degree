from numpy.linalg import inv
import numpy as np
import math


def parameter_estimaion_LR_N(parameter_values):
    follower_orig_log = [parameter_values[i][0] for i in range(len(parameter_values))]
    total_follower_t_log = [math.log(parameter_values[i][1]) for i in range(len(parameter_values))]
    no_events_log = [math.log(parameter_values[i][2]) for i in range(len(parameter_values))]
    event_pred_log = [math.log(parameter_values[i][3]) for i in range(len(parameter_values))]
    n = len(parameter_values)

    follower_orig_sum = sum(follower_orig_log)
    total_follower_t_sum = sum(total_follower_t_log)
    no_events_sum = sum(no_events_log)
    event_pred_sum = sum(event_pred_log)
    a = np.matrix([
        [n, no_events_sum, total_follower_t_sum,
         follower_orig_sum],
        [no_events_sum, no_events_sum * no_events_sum, total_follower_t_sum * no_events_sum,
         follower_orig_sum * no_events_sum],
        [total_follower_t_sum, no_events_sum * total_follower_t_sum, total_follower_t_sum * total_follower_t_sum,
         total_follower_t_sum * follower_orig_sum],
        [follower_orig_sum, no_events_sum * follower_orig_sum, total_follower_t_sum * follower_orig_sum,
         follower_orig_sum * follower_orig_sum]
    ])

    y = (np.matrix([event_pred_sum, no_events_sum * event_pred_sum, total_follower_t_sum * event_pred_sum,
                    follower_orig_sum * event_pred_sum])).transpose()

    theta = inv(a) * y
    theta = np.asarray(theta)
    print(theta)

    sigma_sq = sum(
        [event_pred_log[i] - theta[0] - (theta[1] * no_events_log[i]) - (theta[2] * total_follower_t_log[i]) -
         (theta[3] * follower_orig_log[i]) for i in range(len(event_pred_log))]) / len(event_pred_log)
    print(sigma_sq)

    return theta, sigma_sq


