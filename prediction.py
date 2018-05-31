import math


def prediction_LR_N(parameter_estimmated, paramerer_pred):
    follower_orig_log = [(paramerer_pred[i][0]) for i in range(len(paramerer_pred))]
    total_follower_t_log = [(paramerer_pred[i][1]) for i in range(len(paramerer_pred))]
    no_events_log = [(paramerer_pred[i][2]) for i in range(len(paramerer_pred))]

    r_est = [math.exp(parameter_estimmated[0][0] + (parameter_estimmated[0][1] * no_events_log[i]) +
                      (parameter_estimmated[0][2] *total_follower_t_log[i]) +
                      (parameter_estimmated[0][3] * follower_orig_log[i])+ (parameter_estimmated[1]/2))
             for i in range(len(no_events_log))]

    return r_est