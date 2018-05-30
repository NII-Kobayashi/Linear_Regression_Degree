from function import *
from estimate import *
import glob as gb


file_path = "Data/RT*.txt"
file_list = sorted(gb.glob(file_path), key=numerical_sort) # for files having tweet more than 20000 (RT186. RT1439)
parameters_value = [no_of_events_followers(file_list[i], 6, 10, 3600) for i in range(len(file_list))]
parameters = parameter_estimaion_LR_N(parameters_value)


