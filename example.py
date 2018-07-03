"""
This code trains the linear regression with degree based on a retweet dataset (data/training/RT*.txt),
assuming the parameters are the same in the dataset.
Please replace file paths according to your local directory structure.

Inputs are
1) Data file that includes the retweet times and the number of followers
   Here, this code reads 'Data/training/RT*.txt' (= filename) for training data and 'Data/test/RT*.txt' (= file_name_test) for test data.
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).

Outputs is
1) The estimated parameters (alpha, variance, beta_r, beta_n, beta_0)
2) The prediction result obtained form the model
3) The true prediction value
4) Error estimated

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


# prediction result for one file
file_name_test = "Data/test/RT2560.txt"  # path to the files used for prediction
file_list_test = sorted(gb.glob(file_name_test), key=numerical_sort)  # for all the training file
parameters_value_pred = no_of_events_followers(file_name_test, T_OBS, T_PRED, 3600)

follower_orig_log_test = np.asarray(parameters_value_pred[0])
total_follower_t_log_test = np.asarray(parameters_value_pred[1])
no_events_log_test = np.asarray(parameters_value_pred[2])
event_pred_log_true = np.asarray(parameters_value_pred[3])
event_pred_true = np.asarray(parameters_value_pred[4])
t_prediction_result = prediction_lr_n(parameters_estimated, no_events_log_test, total_follower_t_log_test,
                                           follower_orig_log_test)


error = (abs(event_pred_true - t_prediction_result))
print("The parameters estimated are:")
beta, sigma = parameters_estimated
print("alpha = {0:.3f}".format(round(beta[0][0], 3)))
print("beta_r = {0:.3f}".format(round(beta[0][1], 3)))
print("beta_n = {0:.3f}".format(round(beta[0][2], 3)))
print("beta_0 = {0:.3f}".format(round(beta[0][3], 3)))
print("Variance = {0:.3f}".format(round(sigma[0], 3)))
print("The prediction result for the observation time at ", T_OBS, "hours and the prediction time at", T_PRED,
      "hour is:", int(t_prediction_result[0]))
print("The true value at the prediction time is", event_pred_true)
print("The error estimated is:", int(error))
