import numpy as np
from utils import Task    # 必须要
#
# route = np.load('./route/car_1_route.npy', allow_pickle=True)
# task = np.load('./task/car_1_task.npy', allow_pickle=True) # size/4-8
#
# print(len(task))
# print(len(route))

from utils import plot_img

car_route = np.load('./route/car_1_route.npy', allow_pickle=True)

plot_img(car_route)

