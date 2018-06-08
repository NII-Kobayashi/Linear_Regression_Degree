"""
This code trains the multiple linear regression model parameters (alpha, variance, beta_r, beta_n, beta_0)
based on a re-tweet data-set (data/training/RT*.txt), assuming the parameters are same in the data-set.
Please replace file paths according to your local directory structure.

Inputs are
1) Data file that includes the re-tweet times and the number of followers
Here, this code reads 'Data/training/RT*.txt' (= filename) and 'Data/test/RT*.txt' (= file_name_test) for test data set.
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).

Outputs is
1) the prediction time for the test data set .

This code is developed by Niharika Singhal under the supervision of Ryota Kobayashi.
"""


from function import *
from estimate import *
from prediction import *
import glob as gb
import numpy as np

# estimation
T_OBS = 6
T_PRED = 10
filename = "Data/training/RT*.txt"
file_list = sorted(gb.glob(filename), key=numerical_sort)  # for files having tweet more than 20000 (RT186. RT1439)

parameters_value = [no_of_events_followers(file_list[i], T_OBS, T_PRED, 3600) for i in range(len(file_list))]
parameters_value = list(filter(None.__ne__, parameters_value))  # checking for none value
follower_orig_log = np.asarray([(parameters_value[i][0]) for i in range(len(parameters_value))])
total_follower_t_log = np.asarray([(parameters_value[i][1]) for i in range(len(parameters_value))])
no_events_log = np.asarray([(parameters_value[i][2]) for i in range(len(parameters_value))])
event_pred_log = np.asarray([(parameters_value[i][3]) for i in range(len(parameters_value))])

parameters_estimated = parameter_estimation_lr_n(follower_orig_log, total_follower_t_log, no_events_log, event_pred_log)


# prediction
file_name_test = "Data/test/RT*.txt"  # path to the files used for prediction
file_list_test = sorted(gb.glob(file_name_test), key=numerical_sort)  # for all the training file
parameters_value_pred = [no_of_events_followers(file_list_test[i], T_OBS, T_PRED, 3600)
                         for i in range(len(file_list_test))]
parameters_value_pred = list(filter(None.__ne__, parameters_value_pred))  # checking for none value
follower_orig_log_test = np.asarray([(parameters_value_pred[i][0]) for i in range(len(parameters_value_pred))])
total_follower_t_log_test = np.asarray([(parameters_value_pred[i][1]) for i in range(len(parameters_value_pred))])
no_events_log_test = np.asarray([(parameters_value_pred[i][2]) for i in range(len(parameters_value_pred))])
event_pred_log_true = np.asarray([(parameters_value_pred[i][3]) for i in range(len(parameters_value_pred))])

nfile_prediction_result = [prediction_lr_n(parameters_estimated, no_events_log_test[i], total_follower_t_log_test[i],
                                           follower_orig_log_test[i]) for i in range(len(no_events_log_test))]
print(nfile_prediction_result)



