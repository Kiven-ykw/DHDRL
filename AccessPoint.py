
import numpy as np
import random
import functions as f
from config import Config
from DRQN import drqn
from DQN import DQN
from DQN_for_lower import DQN_lower


class AccessPoint:

    def __init__(self, location, index):

        self.location = location
        self.index = index
        self.flag = 'AP'
        self.n_antennas = Config().n_antennas
        #self.lower_nn = lower_model
        #self.higher_nn = higher_model
        self.lower_nn = drqn()
        self.higher_nn = DQN()
        self.max_power = f.dB2num(Config().ap_power)
        self.search = Config().ap_search_radius
        self.codebook = f.get_codebook()
        self.powerbook = np.arange(Config().n_power_levels) * self.max_power / (Config().n_power_levels - 1)
        self.association_codebook = np.arange(Config().ue_num)
        self.code_index = random.randint(0, Config().codebook_size-1)
        self.power_index = random.randint(0, Config().n_power_levels-1)

        self.association_index = random.randint(0, Config().ue_num-1)

        self.code = self.codebook[:, self.code_index]

        self.power = self.powerbook[self.power_index]

        self.association = self.association_codebook[self.association_index]

        self._init_params_()

    def _init_params_(self):

        self.code_index1, self.code_index2 = None, None
        self.power_index1, self.power_index2 = None, None
        self.association_index1, self.association_index2 = None, None
        self.power1, self.power2 = None, None
        self.association1, self.association2 = None, None

    def _save_low_params_(self):

        self.code_index2 = self.code_index1
        self.code_index1 = self.code_index

        self.power_index2 = self.power_index1
        self.power_index1 = self.power_index

        self.power2 = self.power1
        self.power1 = self.power

    def _save_high_params_(self):

        self.association_index2 = self.association_index1
        self.association_index1 = self.association_index

        self.association2 = self.association1
        self.association = self.association

    def take_low_action(self, action=None, weight=None):
        self._save_low_params_()
        if action is not None:
            self.power_index = action % Config().n_power_levels
            self.code_index = action // Config().n_power_levels
            self.code = self.codebook[:, self.code_index]
            self.power = self.powerbook[self.power_index]
        if weight is not None:
            if np.linalg.norm(weight) != 0:
                self.code = weight / np.linalg.norm(weight)
                self.power = np.square(np.linalg.norm(weight))
            else:
                self.code = np.zeros(Config().n_antennas, dtype=np.complex)
                self.power = 0

    def take_high_action(self, action=None, weight=None):
        self._save_high_params_()
        if action is not None:
            self.association_index = action
            self.association = self.association_codebook[self.association_index]
        if weight is not None:
            if np.linalg.norm(weight) != 0:
                self.code = weight / np.linalg.norm(weight)
                self.power = np.square(np.linalg.norm(weight))
            else:
                self.code = np.zeros(Config().n_antennas, dtype=np.complex)
                self.power = 0


