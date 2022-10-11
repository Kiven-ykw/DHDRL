import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from collections import deque
from config import Config
from tensorflow.python.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

class drqn:
    def __init__(self, num_actions=Config().n_actions, num_observations=8,
                 memory_step=8, memory_episodes=8, lr=0.0005, lr_decay=1e-4,minimum_epsilon=0.01,
                 maximum_epsilon=0.6, epsilon_decay=0.9,target_copy_iterations=100,
                 target_copy_start_steps=10, num_neurons=(64, 32), memory_size=500, future_discount=0.5,
                 learning_rate_decay=1, learning_rate_decay_ep=500, activation_function='relu'):
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.memory_steps = memory_step                  # Number of Steps to Include in an Episode Sample
        self.layer_size_1 = num_neurons[0]
        self.layer_activation = activation_function
        self.layer_size_2 = num_neurons[1]
        self.lr = lr
        self.lr_decay = lr_decay
        self.memory_size = memory_size
        self.learning_rate_decay_ep = learning_rate_decay_ep
        self.memory_episodes = memory_episodes                # Number of Episodes to Include in a Memory Sample
        self.minimum_epsilon = minimum_epsilon
        self.target_copy_iterations = target_copy_iterations
        self.target_copy_start_steps = target_copy_start_steps
        self.future_discount = future_discount
        self.epsilon_decay = epsilon_decay
        self.learning_rate_decay = learning_rate_decay
        self.maximum_epsilon = maximum_epsilon
        self.save_path = 'model/'
        self.optimizer = RMSprop(lr=self.lr, decay=self.lr_decay)

        #Create the model that will be trained
        self.model = Sequential()
        self.model.add(Dense(self.layer_size_1, input_shape=(self.memory_steps, self.num_observations),
                             activation=self.layer_activation))
        self.model.add(LSTM(100, activation=self.layer_activation,unroll=1))
        self.model.add(Dense(self.layer_size_2, activation='linear'))
        self.model.add(Dense(self.num_actions, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

        #Create the model that will calculate target values
        self.model_target = Sequential()
        self.model_target = Sequential.from_config(self.model.get_config())
        self.model_target.set_weights(self.model.get_weights())
        self.model_target.compile(loss='mean_squared_error', optimizer=self.optimizer)

        #Since the LSTM has an internal state that is changed when predictions are made,
        #we keep a separate copy of the training model in order to select actions. This
        #is simpler than tracking the internal state and resetting the values
        self.model_action = Sequential()
        self.model_action = Sequential.from_config(self.model.get_config())
        self.model_action.set_weights(self.model.get_weights())
        self.model_action.compile(loss='mean_squared_error', optimizer=self.optimizer)

        #Create the Replay Memory
        self.replay_memory = deque(maxlen=self.memory_size)
        self.replay_current = []

        self.current_epsilon = self.maximum_epsilon
        self.current_learning_rate = self.lr

        self.train_iterations = 0
        self.first_iterations = 0
        self.current_episode = 0
        self.current_step = 0

        self.average_train_rewards = deque(maxlen=100)
        self.average_test_rewards = deque(maxlen=100)

        # self.train_path = self.save_path + ".train"
        # self.train_file = open(self.train_path, 'w')
        # self.train_file.write('episode reward average_reward\n')
        #
        # self.test_path = self.save_path + ".test"
        # self.test_file = open(self.test_path, 'w')
        # self.test_file.write('episode reward\n')

        #the lstm layer requires inputs of timesteps
        #a history of of past observations are kept to pass into the lstm
        self.lstm_oap_history = deque([[0 for y in range(self.num_observations)] for x in range(self.memory_steps)], self.memory_steps)


    # def __del__(self):
    #     self.train_file.close()
    #     self.test_file.close()
    #     pass
    def load_mod(self, indx):
        self.model = load_model('model/lowap_{}.hdf5'.format(indx))

    def clear_history(self):
        self.lstm_oap_history = deque([[0 for y in range(self.num_observations)] for x in range(self.memory_steps)], self.memory_steps)

    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def choose_action(self, observation):
        #get the current action for the model or a random action depending on arguments
        #and the current episode
        self.lstm_oap_history.append(observation)
        if np.random.random() < self.current_epsilon:
            #if training, choose random actions
            return np.random.randint(0, self.num_actions)
        else:
            #choose action from model
            q_values = self.model_action.predict(np.array(self.lstm_oap_history).reshape(1, self.memory_steps,
                                                                                         self.num_observations))
            action = np.argmax(q_values)
            return action

    def save_transition(self, state, action, reward, next_state):
        #add a transaction to replay memory, should be called after performing
        #an action and getting an observation
        self.replay_current.append((state, action, reward, next_state))
        self.current_step += 1
        #make end of episode checks
        self.replay_memory.append(self.replay_current)
        self.end_of_episode()

    def end_of_episode(self):
        self.current_episode += 1
        self.current_step = 0
        self.clear_history()
        self.replay_current = []
        self.update_action_network()
        self.model_action.reset_states()
        if self.current_epsilon > self.minimum_epsilon:
            self.decay_epsilon()
        else:
            self.current_epsilon = self.minimum_epsilon
        if self.current_episode % self.learning_rate_decay_ep == 0:
            self.decay_learning_rate()

    def sample_memory(self, batch_size, trace_length):
        # samples the replay memory returning a batch_size of random transactions
        sampled_episodes = []
        while True:
            rand_ep = np.random.randint(0, len(self.replay_memory))
            sampled_episodes.append(rand_ep)
            if len(sampled_episodes) == batch_size:
                break
        sampled_traces = []
        for ep in sampled_episodes:
            episode = self.replay_memory[ep]
            start_step = np.random.randint(0, max(1, len(episode) - trace_length + 1))
            current_trace = episode[start_step:start_step + trace_length]
            action = current_trace[-1][1]
            reward = current_trace[-1][2]
            states = []
            next_states = []
            for step, transaction in enumerate(current_trace):
                states.append(transaction[0])
                next_states.append(transaction[3])
            if len(current_trace) < trace_length:
                empty = [0 for x in states[0]]
                for i in range(trace_length - len(current_trace)):
                    states.insert(0, empty)
                    next_states.insert(0, empty)
            sampled_traces.append([states, action, reward, next_states])
        return sampled_traces

    def learn(self):
        if len(self.replay_memory) < self.memory_episodes:
            print('Not enough transactions in replay memory to train.')
            return
        if self.train_iterations >= self.target_copy_iterations:
            self.update_target_network()
        if self.first_iterations < self.target_copy_start_steps:
            # update the target network a few times on episode 0 so
            # the model isn't training toward a completely random network
            self.update_target_network()
            self.first_iterations += 1

        self.model.reset_states()
        self.model_target.reset_states()

        samples = self.sample_memory(self.memory_episodes, self.memory_steps)
        observations = next_observations = rewards = np.array([])
        actions = np.array([], dtype=int)
        for transaction in samples:
            observations = np.append(observations, transaction[0])
            actions = np.append(actions, transaction[1])
            next_observations = np.append(next_observations, transaction[3])
            rewards = np.append(rewards, transaction[2])
        observations = observations.reshape(self.memory_episodes, self.memory_steps, self.num_observations)
        next_observations = next_observations.reshape(self.memory_episodes, self.memory_steps,  self.num_observations)
        targets = updates = None
        if self.target_copy_iterations == 0:
            #this instance is not using a target copy network, use original model
            targets = self.model.predict(observations)
            updates = rewards + self.future_discount * np.max(self.model.predict(next_observations), axis=1)
        else:
            #this instance uses a target copy network
            targets = self.model_target.predict(observations)
            updates = rewards + self.future_discount * np.max(self.model_target(next_observations), axis=1)
        for i, action in enumerate(actions):
            targets[i][action] = updates[i]
        self.model.fit(observations, targets, batch_size=self.memory_episodes, verbose=0)

        self.train_iterations += 1

    def update_target_network(self):
        self.model_target.set_weights(self.model.get_weights())

    def update_action_network(self):
        self.model_action.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        self.current_epsilon *= self.epsilon_decay

    def decay_learning_rate(self):
        self.current_learning_rate *= self.learning_rate_decay

    # def write_training_episode(self, episode, reward):
    #     self.average_train_rewards.append(reward)
    #     self.train_file.write(str(episode) + ' ')
    #     self.train_file.write(str(reward) + ' ')
    #     if len(self.average_train_rewards) >= 100:
    #         self.train_file.write(str(np.mean(self.average_train_rewards)))
    #     self.train_file.write('\n')
    #
    # def write_testing_episode(self, episode, reward):
    #     self.average_test_rewards.append(reward)
    #     self.test_file.write(str(episode) + ' ')
    #     self.test_file.write(str(reward) + ' ')
    #     self.test_file.write('\n')

    def save_model(self, file_name):
        file_path = self.save_path + file_name + '.hdf5'
        self.model.save(file_path, True)