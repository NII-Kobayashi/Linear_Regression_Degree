# Author: Niharika Singhal
#
# For license information, see LICENSE.txt

"""

Full example on how to use linear regression model with degree for predicting re-tweet activity
Please replace file paths according to your local directory structure.

References
----------
.. *Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
"""

from function import *
from estimate import *
from prediction import *
import glob as gb
import numpy as np

# estimation
T_OBS = 6
T_PRED = 10
file_path = "Data/training/RT*.txt"
file_list = sorted(gb.glob(file_path), key=numerical_sort)  # for files having tweet more than 20000 (RT186. RT1439)

parameters_value = [no_of_events_followers(file_list[i], T_OBS, T_PRED, 3600) for i in range(len(file_list))]

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

nfile_prediction_result = [prediction_lr_n(parameters_estimated, no_events_log[i], total_follower_t_log[i],
                                           follower_orig_log[i]) for i in range(len(no_events_log))]
print(nfile_prediction_result)