from numpy.linalg import inv
import numpy as np
from scipy import linalg


def parameter_estimaion_LR_N(follower_orig, total_follower_t, no_events, event_pred, n ):

    follower_orig_sum = sum(follower_orig)
    total_follower_t_sum = sum(total_follower_t)
    no_events_sum = sum(no_events)

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

    y1 = sum(event_pred)
    y2 = sum([(event_pred[i] * event_pred[i]) for i in range(len(event_pred))])
    y3 = sum([(total_follower_t[i] * event_pred[i]) for i in range(len(event_pred))])
    y4 = sum([(follower_orig[i] * event_pred[i]) for i in range(len(event_pred))])

    # theta = inv(a) * y
    # theta = np.asarray(theta)
    # TODO see how to do inverse and correct the result

    if type(y1) == float:
        y = (np.matrix([y1, y2, y3, y4])).transpose()
        theta = linalg.solve(a, y)
        sigma_val = [(event_pred[i] - theta[0] - (theta[1] * no_events[i]) - (theta[2] * total_follower_t[i]) -
                      (theta[3] * follower_orig[i]) ** 2) for i in range(len(event_pred))]
        sigma_sq = sum(sigma_val) / len(event_pred)

        return theta, sigma_sq

    else:
        sigma_sq_list = []
        theta_list = [linalg.solve(a, (np.matrix([y1[i], y2[i], y3[i], y4[i]])).transpose()) for i in range(len(y))]
        for j in range(len(theta_list)):
            sigma_sq = sum([(event_pred[i][j] - theta_list[j][0] - (theta_list[j][1] * no_events[i]) -
                             (theta_list[j][2] * total_follower_t[i]) - (theta_list[j][3] * follower_orig[i]) ** 2)
                            for i in range(len(event_pred))]) / len(event_pred)
            sigma_sq_list.append(sigma_sq)

        return theta_list, sigma_sq_list



