import os
import random
from collections import deque
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')
print(tf.config.list_physical_devices())
print(tf.__version__)

env = gym.make("LunarLander-v2")

SEED = 0


def set_seeds(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)


set_seeds(SEED)


class DQLAgent:
    def __init__(self, env):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.tot_reward = []
        self.batch_size = 64
        self.memory = deque(maxlen=500000)
        self.osn = env.observation_space.shape[0]
        self.opt = Adam(learning_rate=0.001)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.osn, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=self.opt)
        return model

    def act(self, state):
        if random.random() <= self.epsilon:
            return env.action_space.sample()
        action = self.model.predict(state, verbose=0)
        return np.argmax(action[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_batch(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            state = np.squeeze(np.array([i[0] for i in batch]))
            action = np.array([i[1] for i in batch])
            reward = np.array([i[2] for i in batch])
            next_state = np.squeeze(np.array([i[3] for i in batch]))
            done = np.array([i[4] for i in batch])

            target = reward + self.gamma * np.amax(self.model.predict_on_batch(next_state), \
                                                   axis=1) * (1 - done)
            target_f = self.model.predict_on_batch(state)
            idx = np.arange(self.batch_size)
            target_f[[idx], [action]] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            state = env.reset()
            state = np.reshape(state, [1, self.osn])
            t_reward = 0
            max_steps = 1000
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.osn])
                self.memorize(state, action, reward, next_state, done)
                state = next_state
                t_reward += reward
                self.replay_batch()
                if done:
                    print(f'Episode: {e} | Steps: {step} | Total reward: {t_reward} \
                    | Epsilon: {self.epsilon}')
                    break
                    
            self.tot_reward.append(t_reward)

            if e % 10 == 0:
                print(f'Last 10 episodes mean reward: {np.mean(self.tot_reward[-10:])}')

            if np.mean(self.tot_reward[-100:]) >= 200:
                print('==== Environment solved ====')
                break

    def save_model(self):
        self.model.save('LunarLanderDQL')


agent = QLAgent(env)
episodes = 400
agent.learn(episodes)
agent.save_model()

print(max(agent.tot_reward))

df = pd.DataFrame(np.array(agent.tot_reward), columns=['total_reward'])
df['total_reward_50episodes_moving_average'] = df['total_reward'].rolling(window=50).mean()
df.plot(title='Total reward per episode')
plt.xlabel('Episode', fontsize=10)
plt.show()
