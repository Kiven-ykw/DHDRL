import numpy as np
import math
from config import Config
import random


class UserEquipment:

    def __init__(self, location: np.ndarray, index, mobility_range):
        self.location = location
        self.index = index
        self.mobility_range = mobility_range
        self.QoS = 0
        self.require = random.uniform(5, 10)
        self.utility, self.utility11, self.utility12 = 0, 0, 0
        self.r_power, self.r_power11, self.r_power12 = 0, 0, 0

    def __str__(self):
        return "UE" + str(self.index) + "in" + str(self.location) + "with" + str(self.mos)

    @staticmethod
    def limit_user_range(input_shape, shape_range):
        if input_shape[0] < 0:
            input_shape[0] = 0
        elif input_shape[0] >= shape_range[0]:
            input_shape[0] = shape_range[0] - 1
        if input_shape[1] < 0:
            input_shape[1] = 0
        elif input_shape[1] >= shape_range[1]:
            input_shape[1] = shape_range[1] - 1
        return input_shape

    def mobility(self, dis):
        self.location[0] += dis[0]
        self.location[1] += dis[1]
        self.location = self.limit_user_range(self.location, self.mobility_range)

    def require_update(self):
        self.require = np.random.randint(5, 10, 1)

    def location_update(self):
        self.mobility([np.random.randint(0, Config().speed, 1), np.random.randint(0, Config().speed, 1)])

    def dist(self, other):
        return np.sqrt(np.power(self.location[0] - other.location[0], 2) +
                       np.power(self.location[1] - other.location[1], 2))

    def evaluate_MOS(self, utility):
        if utility:
            self.mos = Config().lamda * math.log10(Config().tao * utility)
        else:
            self.mos = 0

