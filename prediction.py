import math


def prediction_lr_n(parameter_estimated, no_events, total_follower_t, follower_orig):

    if len(parameter_estimated[1]) == 1:
        est_rf = math.exp(parameter_estimated[0][0] + (parameter_estimated[0][1] * no_events) +
                           (parameter_estimated[0][2] * total_follower_t) +
                           (parameter_estimated[0][3] * follower_orig) + (parameter_estimated[1] / 2))
        return est_rf

    else:
        est_rf = [math.exp(parameter_estimated[0][j][0] + (parameter_estimated[0][j][1] * no_events)
                           + (parameter_estimated[0][j][2] * total_follower_t)
                           + (parameter_estimated[0][j][3] * follower_orig) + (
                                   parameter_estimated[1][j] / 2))
                  for j in range(len(parameter_estimated[0]))]
        return est_rf
