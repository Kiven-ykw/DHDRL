""" DQN agent at each base station """

from config import Config
import numpy as np
import random
from neural_network import NeuralNetwork
# from neural_network_u import NeuralNetwork_U
from tensorflow.keras.optimizers import RMSprop
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQN:
    # hyper params
    def __init__(self,
                 n_actions=NeuralNetwork().output_ports,
                 n_features=NeuralNetwork().input_ports,
                #  n_actions_u=NeuralNetwork_U().output_ports,
                #  n_features_u=NeuralNetwork_U().input_ports,
                 lr=5e-4,
                 lr_decay=1e-4,
                 reward_decay=0.5,
                 e_greedy=0.6,
                 epsilon_min=1e-2,
                 replace_target_iter=100,
                 memory_size=500,
                 batch_size=8,
                 e_greedy_decay=1e-4):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = reward_decay
        # epsilon-greedy params
        self.epsilon = e_greedy
        self.epsilon_decay = e_greedy_decay
        self.epsilon_min = epsilon_min
        self.save_path = 'model/'

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.loss = []
        self.accuracy = []
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self._built_net()

    def _built_net(self):
        tar_nn = NeuralNetwork()
        eval_nn = NeuralNetwork()
        self.model1 = tar_nn.get_model(1)
        self.model2 = eval_nn.get_model(1)
        self.target_replace_op()

        optimizer = RMSprop(lr=self.lr, decay=self.lr_decay)

        self.model2.compile(loss='mse', optimizer=optimizer)   # 将字符串编译为字节代码
        # self.model4.compile(loss='mse', optimizer=optimizer)   # 将字符串编译为字节代码

    def _store_transition_(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        # print(transition, '\n')
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.array(transition)
        self.memory_counter += 1

    def save_transition(self, s, a, r, s_):
        self._store_transition_(s, a, r, s_)

    def choose_action(self, observation):
        if random.uniform(0, 1) > self.epsilon:
            observations = observation[np.newaxis, :]
            q_values = self.model2.predict(observations)
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # def save(self):
    #     self.model2.save('data/model.h5')
    #     self.model2.save_weights('data/weights.h5')

    def save_model(self, file_name):
        file_path = self.save_path + file_name + '.hdf5'
        self.model2.save(file_path, True)

    def target_replace_op(self):
        temp = self.model2.get_weights()
        print('Parameters updated')
        self.model1.set_weights(temp)

    def learn(self):
        # update target network's params
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()

        # sample mini-batch from experience replay
        if self.memory_counter > self.memory_size:
            sample_index = random.sample(list(range(self.memory_size)), self.batch_size)
        else:
            sample_index = random.sample(list(range(self.memory_counter)), self.batch_size)

        # mini-batch data
        batch_memory = self.memory[sample_index, :]
        q_next = self.model1.predict(batch_memory[:, -self.n_features:])  # 下个状态传给Q目标值网络
        q_eval = self.model2.predict(batch_memory[:, :self.n_features])  # 当前状态传给Q估计值网络
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        hist = self.model2.fit(batch_memory[:, :self.n_features], q_target, verbose=0)
        self.loss.append(hist.history['loss'][0])

        self.epsilon = max(self.epsilon / (1 + self.epsilon_decay), self.epsilon_min)
        self.learn_step_counter += 1

    def save_los(self, i):
        filename = 'Loss/high_ap_{}.json'.format(i)
        with open(filename, 'w') as f:
            json.dump(self.loss, f)

