import functions as f
import numpy as np


class Config:

    def __init__(self):
        self.n_antennas = f.get_codebook().shape[0]  # number of transmit antennas
        self.codebook_size = f.get_codebook().shape[1]  # number of codes
        self.n_power_levels = 5  # number of discrete power levels
        self.n_actions = self.codebook_size * self.n_power_levels   # number of available actions to choose
        self.ap_power = 5.01  # maximum transmit power of base stations
        self.ap_num = 6
        self.mbs_num = 1
        self.ap_access_num = 2  #同时接入AP的数量
        self.ue_num = 3
        self.n_actions_u = self.ue_num #ue的动作个数
        self.ue_mos = 5
        self.lamda = 5
        self.tao = 1
        self.speed = 5
        self.ap_search_radius = 100
        self.negative_reward = -1

        self.reward_decay = 0.99
        self.step_per_association = 10

        # Channel
        self.angular_spread = 3 / 180 * np.pi  # angular spread
        self.multi_paths = 4  # number of multi-paths
        self.rho = 0.35  # channel correlation coefficient
        self.noise_power = f.dB2num(-114)  # noise power

        self.area_size = 600
        self.dense_of_ap = 200
        self.n_links = self.ap_num  # number of simulated direct links in the simulation

        self.slot_interval = 0.02  # interval of one time slot
        self.random_seed = 1996  # random seed to control the simulated cellular network
        self.total_slots = 500000   # total time slots in the simulation
