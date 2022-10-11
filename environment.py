""" simulator for cellular networks """

from AccessPoint import AccessPoint as AP
from user_equipment import UserEquipment as UE
from config import Config
from channel import Channel
import functions as f
import numpy as np
import random

class WirelessNetwork:

    def __init__(self):
        """ initialize the cellular network """
        self.config = Config()
        self._generate_ap_()
        self._generate_ue_()
        self.partner = self.adjacent()
        self._establish_channels_()
        self._reset_()

    def _generate_ap_(self):
        self.ap_list = []
        for ind in range(Config().ap_num):
            self.ap_list.append(AP([np.random.randint(0, Config().area_size), np.random.randint(0, Config().area_size)], ind))

    def _generate_ue_(self):
        self.ue_list = []
        for ind in range(Config().ue_num):
            self.ue_list.append(UE(np.array([np.random.randint(0, Config().area_size), np.random.randint(0, Config().area_size)]), ind,
                                [Config().area_size, Config().area_size]))

    def _establish_channels_(self):
        self.channels = []
        for ap in self.ap_list:
            for ue in self.ue_list:
                self.channels.append(Channel(ap, ue))
        self._get_links_()

    def _update_check_link_(self):
        for channel in self.channels:
            channel._check_is_link_()
        self._get_links_()

    def _get_links_(self):
        self.links = []
        for channel in self.channels:
            if channel.is_link:
                self.links.append(channel)

    def get_channel_list(self, ap_index=None, ue_index=None):
        """ Search for channels that meet the given conditions """
        channel_list = []

        if ap_index is not None and ue_index is None:
            for channel in self.channels:
                if ap_index == channel.ap.index:
                    channel_list.append(channel)
        elif ap_index is None and ue_index is not None:
            for channel in self.channels:
                if ue_index == channel.ue.index:
                    channel_list.append(channel)
        elif ap_index is not None and ue_index is not None:
            for channel in self.channels:
                if ap_index == channel.ap.index and ue_index == channel.ue.index:
                    return channel

        return channel_list

    def get_link_interferers(self, link):
        """ get the set of all the interferers """
        interferers = []
        channels = self.get_channel_list(ue_index=link.ue.index)
        for channel in channels:
            if not channel.is_link:
                interferers.append(channel)
        return interferers

    def _evaluate_link_performance_(self):
        """ evaluate the performance of the link """
        for link in self.links:
            IN = self.config.noise_power
            interferers = self.get_link_interferers(link)
            for interferer in interferers:
                IN += interferer.r_power
            link.IN = IN
            link.SINR = link.r_power / link.IN 
            link.utility = np.log2(1 + link.SINR)
            link.mos = link.evaluate_mos(link.utility)

    def _evaluate_ue_performance_(self):
        """ evaluate the performance of the ue """
        for ue in self.ue_list:
            ue.utility = 0
            ue.utility11 = 0
            for link in self.links:
                if link.index[1] == ue.index:
                    ue.utility += link.utility
                    ue.utility11 += link.utility11
                    ue.r_power += link.r_power

    def update(self, ir_change, flag=None, actions=None, weights=None, ):
        """ update the cellular network status due to channel fading or beamformers update"""
        if ir_change:
            for channel in self.channels:
                channel.update(ir_change)
        else:
            if actions is not None:
                self._take_actions_(flag=flag, actions=actions)
                if flag:
                    self._update_check_link_()
            if weights is not None:
                self._take_actions_(flag=flag, weights=weights)
            # for ue in self.ue_list:
            #     ue.require_update()
            for channel in self.channels:
                channel.update(ir_change)
        self._evaluate_link_performance_()
        self._evaluate_ue_performance_()

    def random_choose_actions(self):
        """ random take actions"""
        actions = []
        for _ in range(self.config.n_links):
            actions.append(random.randint(0, self.config.n_actions - 1))
        return np.array(actions)

    def _take_actions_(self, flag, actions=None, weights=None):
        """ APs take the given actions"""
        """ flag is the index of  high action"""
        if flag is True:
            if actions is not None:
                for index in range(actions.shape[0]):
                    self.ap_list[index].take_high_action(action=actions[index])
        else:
            if actions is not None:
                for index in range(actions.shape[0]):
                    self.ap_list[index].take_low_action(action=actions[index])
            if weights is not None:
                for index in range(weights.shape[1]):
                    self.ap_list[index].take_low_action(weight=weights[:, index])

    def _reset_(self):
        for _ in range(Config().ap_num):
            actions = self.random_choose_actions()
            self.update(ir_change=False, actions=actions)
            self.update(ir_change=True)

    def observe_high(self):
        """ obtain the states of the aps"""
        n_gain = 1e-9
        n_IN = 1e-7
        power_max = f.dB2num(self.config.ap_power)
        n_links = Config().ap_num
        observations = []
        n_position = Config().area_size
        for link in self.links:
            state_relative_position, ue_state = [], []
            local_information = np.hstack((link.ap.power / power_max, link.ap.code_index, link.ue.index,
                                           link.utility11 / n_links,
                                           link.gain / n_gain,
                                           link.IN / n_IN)).tolist()
            for ap in self.ap_list:
                for ue in self.ue_list:
                    state_relative_position.append(self.dist(ap, ue) / n_position)
            #
            for ue in self.ue_list:
                ue_state.append(ue.require)
                ue_state.append(ue.QoS)

            observation = local_information + state_relative_position + ue_state
            observations.append(observation)

        return np.array(observations)

    def observe_low(self, association):
        """ obtain the states of the aps"""
        n_gain = 1e-9
        n_IN = 1e-7
        power_max = f.dB2num(self.config.ap_power)
        n_links = Config().ap_num
        observations = []

        for link in self.links:
            local_information = np.hstack((link.ap.power / power_max, link.ap.code_index,
                                           link.utility11 / n_links, link.gain / n_gain, link.IN / n_IN,
                                           association[link.ap.index][0],
                                           association[link.ap.index][1],
                                           association[link.ap.index][2])).tolist()
            observation = local_information
            observations.append(observation)

        return np.array(observations)

    @staticmethod
    def dist(ap, ue):
        return np.sqrt(np.power(ap.location[0] - ue.location[0], 2) +
                       np.power(ap.location[1] - ue.location[1], 2))

    def adjacent(self):
        adj = []
        for ap1 in self.ap_list:
            sur = []
            for ap in self.ap_list:
                dis = self.dist(ap1, ap)
                if 0 < dis <= ap1.search:
                    sur.append(ap.index)
            adj.append(sur)
        return adj

    def give_high_rewards(self):
        """ calculated the rewards of all the APs in the higher level, conditioned on the current beamforming"""
        rewards = []
        total_reward = 0
        for ue in self.ue_list:
            if ue.utility11 == 0:
                reward = Config().negative_reward
                ue.QoS = 0
            elif ue.utility11 >= ue.require:
                ue.QoS = 1
                reward = 1
            else:
                ue.QoS = 0
                reward = Config().negative_reward
            total_reward = total_reward + reward
        for ap in self.ap_list:
            rewards.append(total_reward/self.config.ue_num)
        return np.array(rewards)

    def give_low_rewards(self):
        """ calculated the rewards of all the APs in the lower level, conditioned on the association"""
        rewards = []
        for link in self.links:
            cluster_reward = link.utility11
            for link_idx in range(0, len(self.partner[link.index[0]])):
                for link1 in self.links:
                    if link1.index[0] == self.partner[link.index[0]][link_idx]:
                        cluster_reward += link1.utility11
            rewards.append(cluster_reward/(len(self.partner[link.index[0]])+1))
        return np.array(rewards)

    def count_ue_qos(self):
        ue_qos = []
        for ue in self.ue_list:
            ue_qos.append(ue.QoS)
        return ue_qos

    def save_high_transitions(self, s, a, r, s_):
        for ap in self.ap_list:
            i = ap.index
            ap.higher_nn.save_transition(s[i, :], a[i], r[i], s_[i, :])

    def save_low_transitions(self, s, a, r, s_):
        for ap in self.ap_list:
            i = ap.index
            ap.lower_nn.save_transition(s[i, :], a[i], r[i], s_[i, :])

    def train_dqns(self, flag=None):
        if flag is True:
            for ap in self.ap_list:
                ap.higher_nn.learn()
        else:
            """ train the DQN of each ap"""
            for ap in self.ap_list:
                ap.lower_nn.learn()

    def choose_actions(self, s, flag=None):
        """ choose actions """
        actions = []
        if flag is True:
            for ap in self.ap_list:
                actions.append(ap.higher_nn.choose_action(s[ap.index, :]))
        else:
            for ap in self.ap_list:
                actions.append(ap.lower_nn.choose_action(s[ap.index, :]))
        return np.array(actions)

    def save_models(self):
        i = 1
        """ save models """
        for ap in self.ap_list:
            ap.lower_nn.save_model('50AP6UE3R100QoSNewRlow_ap_{}'.format(i))
            ap.higher_nn.save_model('50AP6UE3R100QoSNewRhigh_ap_{}'.format(i))
            i += 1

    def save_loss(self):
        i = 1
        for ap in self.ap_list:
            ap.higher_nn.save_los(i)
            i += 1

    def load_models(self):
        i = 1
        for ap in self.ap_list:
            ap.lower_nn.load_mod(i)
            ap.higher_nn.load_mod(i)
            i += 1

    def get_ave_rate(self):
        s = 0
        for ue in self.ue_list:
            s += ue.utility
        return s/self.config.ue_num

    def get_all_rates(self):
        rates = []
        for link in self.links:
            rates.append(link.utility)
        return rates

