"""
This code evaluates the multiple linear regression model parameters, based on a re-tweet data-set
(Data/training/RT*.txt), assuming the parameters are same in the data-set.
Please replace file paths according to your local directory structure.

Input are
1) Data file that includes the re-tweet times and the number of followers
Here, this code reads 'Data/RT*.txt' (= filename) and all the data files are saved in (= file_list_all)
2) Observation time (= obs_time_init).
3) window size to consider multiple prediction time (= window_size).
4) to run for various observation time =(iteration)
5)
6) mean error value obtained form LR model

Output are
1) Errors evaluated via Cross-Validation.
2) plot the mean error at different observation time
3) compare the LR and LR-N model

This code is developed by Niharika Singhal under the supervision of Ryota Kobayashi.
"""

from cross_validation import *
from function import*
import glob as gb
import matplotlib.pyplot as plt


filename = "Data/RT*.txt"
file_list_all = sorted(gb.glob(filename), key=numerical_sort)
obs_time_init = 6  # observation time
window_size = 4  # the window size for multiple prediction value
iteration = 5  # number of time you want to run the loop
mean_list = []  # save the mean value at different observation time
time_list = []  # save the different observation time considered
T_max = 168
for k in range(0, iteration):
    if k == 4:
        obs_time = 72
    else:
        obs_time = obs_time_init * (2 ** k)
    max_value = int((T_max - obs_time) / window_size)
    event_list = [no_of_events_followers_in_window(file_list_all[i], obs_time, window_size, max_value, 3600) for i in
                  range(len(file_list_all))]
    event_list = list(filter(None.__ne__, event_list))  # checking for none value
    result_lr = cross_validation_error(5, event_list, max_value)
    mean_list.append(result_lr[1])
    time_list.append(obs_time)
    print("Time:", obs_time, ", median:", int(result_lr[0]), ", mean:", int(result_lr[1]),
          ", median_corr = {0:.3f}".format(round(result_lr[2], 3)), ", mean_corr = {0:.3f}".format(round(result_lr[3], 3)))

# plot for mean error obtained at different observation time
plt.plot(time_list,  mean_list)
plt.xlabel('T(hour) observation time')
plt.ylabel('Mean absolute error')
plt.xticks(time_list)
plt.ylim(0)
plt.title("Mean Error plot for LR model")
plt.show()

# mean list value from LR model
mean_list_Lr = [2509.7516737808041, 1656.3533519585494, 915.68157126246001, 397.26486540755758, 223.70001051034552]
plt.plot(time_list,  mean_list, time_list,  mean_list_Lr, 'r', linewidth=1.3, alpha=0.8)
plt.xlabel('T(hour) observation time')
plt.ylabel('Mean absolute error')
plt.ylim(0)
plt.title("Comparison of the LR and LR-N value")
plt.legend(['LR-N', 'LR'], loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.show()