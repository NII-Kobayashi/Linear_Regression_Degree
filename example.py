from function import *
from estimate import *
from prediction import *
import glob as gb
from scipy.linalg import solve
# estimation
T_OBS = 6
T_PRED = 10
file_path = "Data/training/RT*.txt"
file_list = sorted(gb.glob(file_path), key=numerical_sort) # for files having tweet more than 20000 (RT186. RT1439)

parameters_value = [no_of_events_followers(file_list[i], T_OBS, T_PRED, 3600) for i in range(len(file_list))]

follower_orig_log = [(parameters_value[i][0]) for i in range(len(parameters_value))]
total_follower_t_log = [(parameters_value[i][1]) for i in range(len(parameters_value))]
no_events_log = [(parameters_value[i][2]) for i in range(len(parameters_value))]
event_pred_log = [(parameters_value[i][3]) for i in range(len(parameters_value))]

# TODO see if above for loops can be removed and some another option to calculate theta the prediction result is wrong
parameters_estimated = parameter_estimaion_LR_N(follower_orig_log, total_follower_t_log,
                                                no_events_log, event_pred_log, len(parameters_value))

# prediction
file_name_test = "Data/test/RT*.txt"  # path to the files used for prediction
file_list_test = sorted(gb.glob(file_name_test), key=numerical_sort)  # for all the training file
parameters_value_pred = [no_of_events_followers(file_list_test[i], T_OBS, T_PRED, 3600) for i in range(len(file_list_test))]
nfile_prediction_result = prediction_LR_N(parameters_estimated, parameters_value_pred)


