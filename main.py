from environment import WirelessNetwork as WN
import json
import random
import numpy as np
from config import Config
import os
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

c = Config()
random.seed(c.random_seed)
np.random.seed(c.random_seed)
WN = WN()
rate = []
mos = []
r_high_last = 0
r_ap_last = 0
test = []
s_high = WN.observe_high()
s_high_last = s_high
actions_high = WN.choose_actions(s_high_last, flag=True)
action_high_convers = np.zeros([Config().ap_num, Config().ue_num])
action_high_convers[np.arange(Config().ap_num), actions_high] = 1
WN.update(ir_change=False, flag=True, actions=actions_high)
r_accumulate = np.zeros(Config().ap_num)
s_low = WN.observe_low(action_high_convers)
step_per_association = Config().step_per_association
ue_QoS = []
high_reward = []
count_high_action = []
count_beam_action = []
count_power_action = []
for _ in range(c.total_slots):
    print(_)
    if _ % step_per_association == 0:
        """Performing user association in higher level"""
        r_discounted = r_accumulate * (Config().reward_decay ** Config().step_per_association)
        WN.save_high_transitions(s_high_last, actions_high, r_discounted, s_high)
        actions_high = WN.choose_actions(s_high, flag=True)
        WN.update(ir_change=False,  flag=True, actions=actions_high)
        action_high_convers = np.zeros([Config().ap_num, Config().ue_num])
        action_high_convers[np.arange(Config().ap_num), actions_high] = 1

        s_high_last = s_high
        r_accumulate = np.zeros(Config().ap_num)

        count_temp = []
        for link in WN.links:
            count_temp.append(link.ap.association)
        count_high_action.append(count_temp)

        if _ > 256:
            WN.train_dqns(True)

    """Performing beamforming and power allocation in lower level, conditioned on the association"""
    s_low = WN.observe_low(action_high_convers)
    actions = WN.choose_actions(s_low, flag=False)
    WN.update(ir_change=False, flag=False, actions=actions)
    rate.append(WN.get_ave_rate())
    ue_QoS.append(WN.count_ue_qos())
    high_reward.append(WN.give_high_rewards())
    WN.update(ir_change=True)

    r_high = WN.give_high_rewards()
    r_ap = WN.give_low_rewards()
    r_ap_new = r_ap - r_ap_last

    s_low_next = WN.observe_low(action_high_convers)

    WN.save_low_transitions(s_low, actions, r_ap_new, s_low_next)

    s_high = WN.observe_high()

    r_ap_last = r_ap
    r_accumulate += r_high

    count_beam_temp = []
    count_power_temp = []
    for link in WN.links:
        count_beam_temp.append(link.ap.code_index)
        count_power_temp.append(link.ap.power_index)
    count_beam_action.append(count_beam_temp)
    count_power_action.append(count_power_temp)

    if _ > 256:
        WN.train_dqns(False)
WN.save_models()

filename = 'RatevsTimeslot/rate500000R100rho35AP6UE3QoS.json'
with open(filename, 'w') as f:
    json.dump(rate, f)

QoS_m = np.array(ue_QoS)
sio.savemat('QoS/proposedR100rho35QoS.mat', {'proposedR100rho35QoS': QoS_m})

high_reward_m = np.array(high_reward)
sio.savemat('QoS/proposedR100rho35HR.mat', {'proposedR100rho35HR': high_reward_m})

high_action_m = np.array(count_high_action)
sio.savemat('QoS/proposedR100rho35ASS.mat', {'proposedR100rho35ASS': high_action_m})

beam_m = np.array(count_beam_action)
sio.savemat('QoS/proposedR100rho35Beam.mat', {'proposedR100rho35Beam': beam_m})

power_m = np.array(count_power_action)
sio.savemat('QoS/proposedR100rho35Pow.mat', {'proposedR100rho35Pow': power_m})




