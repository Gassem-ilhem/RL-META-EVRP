import numpy as np
import tensorflow as tf

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size 
        self.memory = []
        self.memory_size = memory_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return np.random.randint(self.action_size)
    #     q_values = self.model.predict(state)
    #     return np.argmax(q_values[0])
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        states = []
        targets = []
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return
    #     minibatch = np.array(self.memory)[np.random.choice(len(self.memory), size=batch_size, replace=False)]
    #     states = np.array([m[0] for m in minibatch])
    #     actions = np.array([m[1] for m in minibatch])
    #     rewards = np.array([m[2] for m in minibatch])
    #     next_states = np.array([m[3] for m in minibatch])
    #     dones = np.array([m[4] for m in minibatch])
    #     targets = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) * (1 - dones)
    #     target_f = self.model.predict(states)
    #     target_f[np.arange(len(states)), actions] = targets
    #     self.model.fit(states, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
























# import numpy as np
# import tensorflow as tf

# class DQNAgent:
#     def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.memory = []
#         self.memory_size = memory_size
#         self.model = self._build_model()

#     def _build_model(self):
#         model = tf.keras.Sequential()
#         model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(tf.keras.layers.Dense(24, activation='relu'))
#         model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         if len(self.memory) >= self.memory_size:
#             self.memory.pop(0)
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return np.random.randint(self.action_size)
#         q_values = self.model.predict(state)
#         return np.argmax(q_values[0])

#     def replay(self, batch_size):
#         if len(self.memory) < batch_size:
#             return
#         minibatch = np.array(self.memory)[np.random.choice(len(self.memory), size=batch_size, replace=False)]
#         states = np.array([m[0] for m in minibatch])
#         actions = np.array([m[1] for m in minibatch])
#         rewards = np.array([m[2] for m in minibatch])
#         next_states = np.array([m[3] for m in minibatch])
#         dones = np.array([m[4] for m in minibatch])
#         targets = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) * (1 - dones)
#         target_f = self.model.predict(states)
#         target_f[np.arange(len(states)), actions] = targets
#         self.model.fit(states, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay