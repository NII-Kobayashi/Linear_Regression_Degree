# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""
Implements functions to check the accuracy of the model by using cross- validation

References
----------
.. *Szabo and Huberman, Communication of the ACM 53, 80 2010; Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""

import numpy as np
from sklearn.model_selection import KFold
import statistics
from estimate import *
from prediction import *


def cross_validation_error(k_fold, event_list_data, max_value_itr):
    """
       estimate the mean, media error and mean, median correlation
       :param k_fold: the number of times we want to use the cross-validation
       :param event_list_data: list log and anti log values, for the re-tweet at time t and re-tweet at multiple
       prediction time, from all the data files
       :return: the average mean, media error and correlation
       """
    parameters_value_list = []
    r_inf_estimated_list = []
    est_list_all = []
    actual_list_all = []
    error_list_all = []
    correlation_list = []
    kf = KFold(n_splits=k_fold)
    for train_index, test_index in kf.split(event_list_data):

        original_followers = [(event_list_data[i][0]) for i in range(len(event_list_data))]
        total_follower = [(event_list_data[i][1]) for i in range(len(event_list_data))]
        event_t = [(event_list_data[i][2]) for i in range(len(event_list_data))]

        event_pred_actual = [(event_list_data[i][3]) for i in range(len(event_list_data))]

        original_followers_arr = np.asarray(original_followers)
        total_follower_arr = np.asarray(total_follower)
        event_t_arr = np.asarray(event_t)
        event_pred_actual_arr = np.asarray(event_pred_actual)

        train_original_foll, test_original_foll = original_followers_arr[train_index], original_followers_arr[
            test_index]
        train_tot_foll, test_tot_foll = total_follower_arr[train_index], total_follower_arr[test_index]
        train_event_t, test_event_t = event_t_arr[train_index], event_t_arr[test_index]
        train_event_pred_act, test_event_pred_act = event_pred_actual_arr[train_index], event_pred_actual_arr[
            test_index]

        parameters_estimated = parameter_estimaion_LR_N(train_original_foll, train_tot_foll,
                                                         train_event_t, train_event_pred_act, len(train_original_foll))

        # TODO prediction but first check the estimation result
