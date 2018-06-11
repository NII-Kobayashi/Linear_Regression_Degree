"""
This code plot the mean error and the variance for the simple linear regression model
based on a re-tweet data-set (data/training/RT*.txt), assuming the parameters are same in the data-set.
Please replace file paths according to your local directory structure.
It also compare the result obtained from LR and LR-N. The result from the LR is saved in the variable =(mean_lr)

Inputs are
1) Data file that includes the re-tweet times and the number of followers
Here, this code reads 'Data/training/RT*.txt' (= filename) and 'Data/test/RT*.txt' (= file_name_test) for test data set.
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).
4) runtime for the T_OBS interval of 6 hours

Outputs is
1) the plot shows the comparison between LR and LR-N.
2) the plot showing mean error and the variance.

This code is developed by Niharika Singhal under the supervision of Ryota Kobayashi.
"""
from function import *
from estimate import *
from prediction import *
import glob as gb
import numpy as np
import matplotlib.pyplot as plt


def mean_error_sd(t_obs, t_pred, file_list_train, file_list_test_):
    parameters_value = [no_of_events_followers(file_list_train[i], t_obs, t_pred, 3600) for i in range(len(file_list_train))]
    parameters_value = list(filter(None.__ne__, parameters_value))  # checking for none value
    follower_orig_log = np.asarray([(parameters_value[i][0]) for i in range(len(parameters_value))])
    total_follower_t_log = np.asarray([(parameters_value[i][1]) for i in range(len(parameters_value))])
    no_events_log = np.asarray([(parameters_value[i][2]) for i in range(len(parameters_value))])
    event_pred_log = np.asarray([(parameters_value[i][3]) for i in range(len(parameters_value))])

    parameters_estimated = parameter_estimation_lr_n(follower_orig_log, total_follower_t_log, no_events_log,
                                                     event_pred_log)
    # prediction
    parameters_value_pred = [no_of_events_followers(file_list_test_[i], t_obs, t_pred, 3600)
                             for i in range(len(file_list_test_))]
    parameters_value_pred = list(filter(None.__ne__, parameters_value_pred))  # checking for none value
    follower_orig_log_test = np.asarray([(parameters_value_pred[i][0]) for i in range(len(parameters_value_pred))])
    total_follower_t_log_test = np.asarray([(parameters_value_pred[i][1]) for i in range(len(parameters_value_pred))])
    no_events_log_test = np.asarray([(parameters_value_pred[i][2]) for i in range(len(parameters_value_pred))])
    event_pred_log_true = np.asarray([(parameters_value_pred[i][3]) for i in range(len(parameters_value_pred))])
    event_pred_true = np.asarray([(parameters_value_pred[i][4]) for i in range(len(parameters_value_pred))])
    nfile_prediction_result = [
        prediction_lr_n(parameters_estimated, no_events_log_test[i], total_follower_t_log_test[i],
                        follower_orig_log_test[i]) for i in range(len(no_events_log_test))]
    error = [(abs(event_pred_true[i] - nfile_prediction_result[i])) for i in range(len(event_pred_true))]
    mean_error = np.mean(error)
    sd = np.std(error)
    return mean_error, sd


filename = "Data/training/RT*.txt"
file_list = sorted(gb.glob(filename), key=numerical_sort)  # for files having tweet more than 20000 (RT186. RT1439)
file_name_test = "Data/test/RT*.txt"  # path to the files used for prediction
file_list_test = sorted(gb.glob(file_name_test), key=numerical_sort)  # for all the training file


# plot at different observation time and fixed prediction time
T_OBS = 6
T_PRE = 78
runtime = 13
res = [mean_error_sd((T_OBS*j), T_PRE, file_list, file_list_test) for j in range(1, runtime)]
mean_lrn = [res[i][0] for i in range(len(res))]
std_lrn = [res[i][1] for i in range(len(res))]

# mean_lr value obtained after running linear regression model
mean_lr = [1792.0844759662882, 1211.2897546952167, 894.82557356704251, 642.7528622049914, 461.42831410836766,
           381.45679911941625, 295.03523547681004, 186.20665950234905, 130.54609867826824, 105.31366309764059,
           78.501781006656799, 36.473511541637663]

tx = np.arange(T_OBS, T_OBS*runtime,T_OBS)
plt.plot(tx,  np.log(mean_lrn), tx,  np.log(mean_lr), 'r', linewidth=1.3, alpha=0.8)
plt.xlabel('T(hour) observation time')
plt.ylabel('Log Mean absolute error')
plt.xticks(np.arange(T_OBS, T_OBS*runtime, T_OBS))
plt.ylim(0)
plt.title("Prediction value at T = 78 hours for different observation time")
plt.legend(['LR-N', 'LR'], loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.show()

# variance and mean error plot for LR-N
plt.errorbar(np.arange(T_OBS, T_OBS*runtime, T_OBS), np.log(mean_lrn), np.log(std_lrn), linestyle='None', marker='^', capsize=3)
plt.xlabel('T(hour) observation time')
plt.ylabel('Log Mean error')
plt.ylim(0)
plt.xticks(np.arange(T_OBS, T_OBS*runtime, T_OBS))
plt.title("Prediction value at T = 78 hours for different observation time")
plt.grid(True)
plt.show()


