# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Full example to checks the accuracy of the model.
Please replace file paths according to your local directory structure.

References
----------
.. *Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""

from cross_validation import *
from function import*
import glob as gb


def main(obs_time, file_list):
    """
    print the mean, median and correlation error at the observation
    :param obs_time: observation time
    :param file_list: data files
    """
    for q in range(0, 6):
        if q == 4:
            dt = 24
        if q == 5:
            dt = 162
        else:
            dt = (2 ** q)
        max_value = int((168 - obs_time) / dt) # number of bins
        window_size = dt  # dt in second
        event_list = [no_of_events_followers_in_window(file_list[i], obs_time, window_size, max_value, 3600) for i in
                      range(len(file_list))]
        event_list = list(filter(None.__ne__, event_list))  # checking for none value
        result_lr = cross_validation_error(5, event_list, max_value)
        print("Time:", dt, result_lr)


file_path = "Data/RT*.txt"
file_list_all = sorted(gb.glob(file_path), key=numerical_sort)
main(6, file_list_all)


# creating the code
max_value = int((168 - 6) / 2)  # number of bins
window_size = 2  # dt in second
event_list = [no_of_events_followers_in_window(file_list_all[i], 6, window_size, max_value, 3600) for i in
              range(len(file_list_all))]
event_list = list(filter(None.__ne__, event_list))
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(event_list):

    original_followers = [(event_list[i][0]) for i in range(len(event_list))]
    total_follower = [(event_list[i][1]) for i in range(len(event_list))]
    event_t = [(event_list[i][2]) for i in range(len(event_list))]
    event_pred_actual = [(event_list[i][3]) for i in range(len(event_list))]

    original_followers_arr = np.asarray(original_followers)
    total_follower_arr = np.asarray(total_follower)
    event_t_arr = np.asarray(event_t)
    event_pred_actual_arr = np.asarray(event_pred_actual)

    train_original_foll, test_original_foll = original_followers_arr[train_index], original_followers_arr[test_index]
    train_tot_foll, test_tot_foll = total_follower_arr[train_index], total_follower_arr[test_index]
    train_event_t, test_event_t = event_t_arr[train_index], event_t_arr[test_index]
    train_event_pred_act, test_event_pred_act = event_pred_actual_arr[train_index], event_pred_actual_arr[test_index]

    #parameters_estimated = parameter_estimaion_LR_N(train_original_foll, train_tot_foll,
    # train_event_t, train_event_pred_act, len(train_original_foll))

    follower_orig_sum = sum(train_original_foll)
    total_follower_t_sum = sum(train_tot_foll)
    no_events_sum = sum(train_event_t)

    a = np.matrix([
        [len(train_original_foll), no_events_sum, total_follower_t_sum,
         follower_orig_sum],
        [no_events_sum, no_events_sum * no_events_sum, total_follower_t_sum * no_events_sum,
         follower_orig_sum * no_events_sum],
        [total_follower_t_sum, no_events_sum * total_follower_t_sum, total_follower_t_sum * total_follower_t_sum,
         total_follower_t_sum * follower_orig_sum],
        [follower_orig_sum, no_events_sum * follower_orig_sum, total_follower_t_sum * follower_orig_sum,
         follower_orig_sum * follower_orig_sum]
    ])

    print(a)

    y1 = sum(train_event_pred_act)
    y2 = sum([(train_event_t[i] * train_event_pred_act[i]) for i in range(len(train_event_pred_act))])
    y3 = sum([(train_tot_foll[i] * train_event_pred_act[i]) for i in range(len(train_event_pred_act))])
    y4 = sum([(train_original_foll[i] * train_event_pred_act[i]) for i in range(len(train_event_pred_act))])
    y = (np.matrix([y1, y2, y3, y4])).transpose()

    theta_list = [linalg.solve(a, (np.matrix([y1[i], y2[i], y3[i], y4[i]])).transpose()) for i in range(len(y))]

    sigma_sq_list = []
    for j in range(len(theta_list)):
        sigma_sq = sum([(train_event_pred_act[i][j] - theta_list[j][0] - (theta_list[j][1] * train_event_t[i]) -
                         (theta_list[j][2] * train_tot_foll[i]) -(theta_list[j][3] * train_original_foll[i]) ** 2)
                            for i in range(len(train_event_pred_act))]) / len(train_event_pred_act)
        print(sigma_sq)
        sigma_sq_list.append(sigma_sq)


    # prediction


