"""
This code plots the mean error and the variance for the simple linear regression model
based on a retweet dataset (data/training/RT*.txt), assuming the parameters are the same for all the tweets.
Please replace file paths according to your local directory structure.
It also compare the result obtained from linear regression and from linear regression with degree. The result of the linear regression is saved in the variable =(mean_lr)

Inputs are
1) Data file that includes the retweet times and the number of followers
   (Explain me!!) Here, this code reads 'Data/training/RT*.txt' (= filename) and 'Data/test/RT*.txt' (= file_name_test) for test data set. (Explain me!!)
2) Observation time (= T_OBS).
3) Final time of prediction (= T_PRED).
4) (Explain me!!) Runtime for the T_OBS interval of 6 hours (Explain me!!)
5) Mean error value obtained from LR model

Outputs is
1) The plot showing mean error and the variance.
2) The plot shows the comparison between LR and LR-N.

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
    quan1 = np.percentile(error, 25)
    quan3 = np.percentile(error, 75)
    return mean_error, sd, quan1, quan3



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
q1 = [res[i][2] for i in range(len(res))]
q3 = [res[i][3] for i in range(len(res))]


# variance and mean error plot for LR-N
plt.errorbar(np.arange(T_OBS, T_OBS*runtime, T_OBS), mean_lrn, std_lrn, linestyle='None', marker='^', capsize=3)
plt.plot(np.arange(T_OBS, T_OBS*runtime, T_OBS), q1, 'k_')
plt.plot(np.arange(T_OBS, T_OBS*runtime, T_OBS), q3, 'k_')
plt.xlabel('T(hour) observation time')
plt.ylabel('Mean error')
plt.xticks(np.arange(T_OBS, T_OBS*runtime, T_OBS))
plt.title("Prediction value at T = 78 hours for different observation time")
plt.show()


# mean_lr value obtained after running linear regression model
mean_lr = [1792.0844759662882, 1211.2897546952167, 894.82557356704251, 642.7528622049914, 461.42831410836766,
           381.45679911941625, 295.03523547681004, 186.20665950234905, 130.54609867826824, 105.31366309764059,
           78.501781006656799, 36.473511541637663]


tx = np.arange(T_OBS, T_OBS*runtime,T_OBS)
plt.plot(tx,  mean_lrn, tx,  mean_lr, 'r', linewidth=1.3, alpha=0.8)
plt.xlabel('T(hour) observation time')
plt.ylabel('Mean absolute error')
plt.xticks(np.arange(T_OBS, T_OBS*runtime, T_OBS))
plt.ylim(0)
plt.title("Prediction value at T = 78 hours for different observation time")
plt.legend(['LR-N', 'LR'], loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.show()
