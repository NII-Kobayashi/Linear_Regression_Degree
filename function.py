import re
import warnings

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    """
    numerically sort the filename path in the directory.
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def no_of_events_followers(event_file, t_observation, t_prediction, time_factor=1):
    """
    calculate the tweet number
    :param event_file: path to file
    :param t_prediction: prediction time
    :param time_factor: factor to multiply time with, useful to convert time unit
    :return: original number of follower, number of event until observation time,
    """
    event_no_t_obs = 0
    follower_t = 0  # will store number of follower before observation time
    original_follower = 0
    event_t_pred = 0
    with open(event_file, "r") as in_file:
        first = next(in_file)  # to remove the first line in the tweet file
        for num, line in enumerate(in_file, 1):
            values = line.split(" ")
            if num == 1:
                original_follower = int(values[1])
            if float(values[0]) <= (t_observation * time_factor):
                follower_t = follower_t + int(values[1])
                event_no_t_obs = num
            if float(values[0]) <= (t_prediction * time_factor):
                event_t_pred = num

    if event_no_t_obs == 0:
        warnings.warn("No event have occurred till the observation time. The file WILL BE IGNORED")
        print("Ignored File Name:", event_file)
    else:
        return original_follower, follower_t, event_no_t_obs, event_t_pred


def no_of_events_followers_in_window(event_file, t_observation, win_size, max_itr, time_factor=1):
    """
    calculate the total number of re-tweet at the observation time and the multiple prediction time
    :param event_file: data file
    :param t_observation: observation time
    :param time_factor: factor to multiply time to convert the time unit in seconds
    :param win_size: the window size for multiple prediction value
    :param max_itr: define the iteration for the window size
    :return: total no of re-tweets at observation time, multiple prediction value
    """
    event_no_t_obs = 0
    event_no_t_obs_list =[]
    follower_t = 0
    follower_t_list = []
    original_follower = 0
    original_follower_list =[]
    event_t_pred = 0
    event_t_pred_list =[]
    time = t_observation * time_factor  # converting in seconds
    win_size_sec = win_size * time_factor
    t_f_list = [time + (i * win_size_sec) for i in range(1, max_itr + 1)]

    for i in range(len(t_f_list)):
        with open(event_file, "r") as in_file:
            first = next(in_file)  # to remove the first line containing total no of re-tweet and no of follower
            for num, line in enumerate(in_file, 1):
                values = line.split(" ")
                if num == 1:
                    original_follower = int(values[1])
                if float(values[0]) <= time:
                    follower_t = follower_t + int(values[1])
                    event_no_t_obs = num
                if float(values[0]) <= (t_f_list[i] * time_factor):
                    event_t_pred = num

            if event_no_t_obs == 0:  # ignoring the file if there is no event happened during the observation time
                warnings.warn("No event have occurred till the observation time. The file WILL BE IGNORED")
                print("Ignored File Name:", event_file)
            else:
                original_follower_list.append(original_follower)
                follower_t_list.append(follower_t)
                event_no_t_obs_list.append(event_no_t_obs)
                event_t_pred_list.append(event_t_pred)

    return original_follower_list, follower_t_list, event_no_t_obs_list, event_t_pred_list  #