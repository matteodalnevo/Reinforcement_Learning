# %% [markdown]
# Questa versione fa A2C - CNN

# %% [markdown]
# # Snake

# %%
import environments_fully_observable
import environments_partially_observable
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

# %% [markdown]
# ## Environment definition

# %%
size = 7
# %matplotlib inline
# function to standardize getting an env for the whole notebook
def get_env(n=1000):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    e = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    # or environments_partially_observable.OriginalSnakeEnvironment(n, size, 2)
    return e

# %% [markdown]
# # **A2C**

# %%
class A2C(tf.keras.Model):
    def __init__(self, n_actions):
        super(A2C, self).__init__()
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(7, 7, 1), padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(n_actions, activation='softmax')
        ])
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(7, 7, 1), padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ])


# %%
# Example of usage
env = get_env(1)
action_dim = 4  # Dimensionality of the action space

# Create an instance of the QN model
a2c_net = A2C(action_dim)
# Compile the model
a2c_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()],
                 metrics=[['accuracy'], ['accuracy']])

# Example usage: obtain Q-values for a state
state = env.boards / 4.0
action_probs = a2c_net.actor(state)
critic_value = a2c_net.critic(state)

print("Action probabilities:", action_probs)
print("Critic Value:", critic_value)

action = tf.random.categorical(logits=action_probs, num_samples=1)
print("Selected action",action)

# %% [markdown]
# # Collect one step in the environments

# %%

def collect_one_step(envs, a2c_net,TEST):
    """
    Collects one transitions S,A,R,S'

    Args:
    - envs: The environments.
    - a2c: The NN.

    Returns:
    - states, actions, rewards, next_states
    """

    # Collect the states
    states = envs.boards / 4.0
    #print(envs.boards[0])
    # Compute the actions probabilities
    actions_probs = a2c_net.actor(states)
    #print("act prob:",actions_probs)
    # Sample from them
    if TEST:
        # Find the index of the action with the maximum probability for each element in the batch
        max_indices = tf.argmax(actions_probs, axis=1)
        # Reshape max_indices
        actions = tf.reshape(max_indices, (-1, 1))
    else:
        actions = tf.random.categorical(tf.math.log(actions_probs), 1)

    # Move in the environment and collect the rewards
    rewards = envs.move(actions)
    # Collect next states
    next_states = envs.boards / 4.0

    return states, actions, rewards, next_states


# %%
# Example of usage
# Initialize the environment
envs = get_env(256)
action_dim = 4  # Dimensionality of the action space

# Create an instance of the QN model
a2c_net = A2C(action_dim)

TEST = True

states, actions, rewards, next_states = collect_one_step(envs, a2c_net,TEST)

print(states.shape, actions.shape, rewards.shape, next_states.shape)

# %%
input_shape = envs.boards.shape
print(input_shape)
# a2c_net.critic.build(input_shape=input_shape)
a2c_net.critic.summary()
from tensorflow.keras.utils import plot_model
plot_model(a2c_net.critic, to_file='model.png', show_shapes=True)

# %%
def get_length(data, total_envs):
  # Get unique board numbers and their counts
  board_numbers, counts = np.unique(data[:, 0], return_counts=True)

  # Create a tensor with all board numbers and initial count of 1
  lengths = np.ones((total_envs, 1), dtype=int)

  # Update the tensor with counts from the data
  for board_number, count in zip(board_numbers, counts):
      lengths[int(board_number) - 1] = count +1

  return lengths

# %%
def plotting(max_steps, n_fruit_eaten, n_wall_hit, n_body_hit, rewards):
  # Create a figure and axis objects
  fig, axs = plt.subplots(1, 4, figsize=(15, 5))

  # Plot n_fruit_eaten
  axs[0].plot(range(1, max_steps + 1), n_fruit_eaten)
  axs[0].set_xlabel('Training Steps')
  axs[0].set_ylabel('Eaten fruits')
  axs[0].set_title('Eaten fruits over Training Steps')
  axs[0].grid(True)

  # Plot n_wall_hit
  axs[1].plot(range(1, max_steps + 1), n_wall_hit)
  axs[1].set_xlabel('Training Steps')
  axs[1].set_ylabel('Wall hit')
  axs[1].set_title('Wall hit over Training Steps')
  axs[1].grid(True)

  # Plot n_body_hit
  axs[2].plot(range(1, max_steps + 1), n_body_hit)
  axs[2].set_xlabel('Training Steps')
  axs[2].set_ylabel('Body hit')
  axs[2].set_title('Body hit over Training Steps')
  axs[2].grid(True)

  # Plot n_fruit_eaten
  axs[3].plot(range(1, max_steps + 1), rewards)
  axs[3].set_xlabel('Training Steps')
  axs[3].set_ylabel('Rewards')
  axs[3].set_title('Rewards over Training Steps')
  axs[3].grid(True)

  # Adjust layout
  plt.tight_layout()

  # Show the plot
  plt.show()

# %%
def test(envs, a2c_net, max_steps, TEST):

  rewardsF = []
  fruit_data = []
  head_data = []
  body_data = []
  n_fruit_eaten = []
  n_wall_hit = []
  n_body_hit = []

  for step in trange(1, max_steps + 1):

      # Do one step in the environment

      states, actions, rewards, next_states = collect_one_step(envs, a2c_net, TEST)

      # Save the data of the first board
      heads = np.argwhere(envs.boards == envs.HEAD) # find the head in the board for all the games
      fruits = np.argwhere(envs.boards == envs.FRUIT) # find the fruit in the board for all the games
      bodies = np.argwhere(envs.boards == envs.BODY)
      filtered_body = [[row[1], row[2]] for row in bodies if row[0] == 0]

      fruit_data.append([fruits[0][1],fruits[0][2]])
      head_data.append([heads[0][1],heads[0][2]])
      body_data.append(filtered_body)

      n_fruit_eaten.append(np.count_nonzero(np.array(rewards) == .5))
      n_wall_hit.append(np.count_nonzero(np.array(rewards) == -.1))
      n_body_hit.append(np.count_nonzero(np.array(rewards) == -.2))

      rewardsF.append(np.mean(np.array(rewards)))

  print("Test is completed!")

  return rewardsF, n_fruit_eaten, n_wall_hit, n_body_hit, fruit_data, head_data, body_data

# %% [markdown]
# # Test

# %%
# Initialize the environment
envs = get_env(256)
action_dim = 4  # Dimensionality of the action space

# Create an instance of the Net model
a2c_net_test = A2C(action_dim)
# Create optimizer for actor and critic separately
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Compile the model
a2c_net_test.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()],
                 metrics=[['accuracy'], ['accuracy']])

# Call the model with some dummy input to initialize the weights
a2c_net_test.critic(states)
a2c_net_test.actor(states)

a2c_net_test.load_weights('./weights/weights_a2c_cnn.weights.h5')

max_steps = int(20_000)

TEST = True

rewards, n_fruit_eaten, n_wall_hit, n_body_hit, fruit_data, head_data, body_data = test(envs, a2c_net_test, max_steps, TEST)

# plotting(max_steps, n_fruit_eaten, n_wall_hit, n_body_hit, rewards)

rewards_ep = np.mean(np.reshape(rewards, (100, 200)), axis=1)
n_fruit_eaten_ep = np.mean(np.reshape(n_fruit_eaten, (100, 200)), axis=1)
n_wall_hit_ep = np.mean(np.reshape(n_wall_hit, (100, 200)), axis=1)
n_body_hit_ep = np.mean(np.reshape(n_body_hit, (100, 200)), axis=1)
max_steps = rewards_ep.shape[0]

plotting(max_steps, n_fruit_eaten_ep, n_wall_hit_ep, n_body_hit_ep, rewards_ep)