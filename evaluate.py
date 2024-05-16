# %% [markdown]
# Test the DDQN

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
#%matplotlib inline
# function to standardize getting an env for the whole notebook
def get_env(n=1000):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    e = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    # or environments_partially_observable.OriginalSnakeEnvironment(n, size, 2)
    return e

# %% [markdown]
# # **DDQN**

# %%
class DDQN(tf.keras.Model):
    def __init__(self, action_dim):
        super(DDQN, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_dim)
        ])

# %% [markdown]
# # Collect one step in the environments

# %%
def collect_one_step(envs, q_net, epsilon):
    """
    Collects one transitions S,A,R,S'

    Args:
    - envs: The environments.
    - a2c: The NN.

    Returns:
    - states, actions, rewards, next_states
    """

    states = envs.to_state()

    q_values = q_net.model(states)

    if np.random.rand() < epsilon:
      actions = tf.random.categorical(q_values, 1)
    else:
      actions = tf.expand_dims(tf.argmax(q_values, axis=1), axis=1)

    # Observe rewards
    rewards = envs.move(actions)

    next_states = envs.to_state()

    return states, actions, rewards, next_states

# %% [markdown]
# # Get Lengths of snake

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

# %% [markdown]
# # Plot

# %%
def plotting(max_steps, n_fruit_eaten, n_wall_hit, n_body_hit, rewards):
  # Create a figure and axis objects
  fig, axs = plt.subplots(1, 4, figsize=(15, 5))

  # Plot n_fruit_eaten
  axs[0].plot(range(1, max_steps + 1), n_fruit_eaten)
  axs[0].set_xlabel('Episodes')
  axs[0].set_ylabel('Eaten fruits')
  axs[0].set_title('Eaten fruits over Episodes')

  # Plot n_wall_hit
  axs[1].plot(range(1, max_steps + 1), n_wall_hit)
  axs[1].set_xlabel('Episodes')
  axs[1].set_ylabel('Wall hit')
  axs[1].set_title('Wall hit over Episodes')

  # Plot n_body_hit
  axs[2].plot(range(1, max_steps + 1), n_body_hit)
  axs[2].set_xlabel('Episodes')
  axs[2].set_ylabel('Body hit')
  axs[2].set_title('Body hit over Episodes')

  # Plot n_fruit_eaten
  axs[3].plot(range(1, max_steps + 1), rewards)
  axs[3].set_xlabel('Episodes')
  axs[3].set_ylabel('Rewards')
  axs[3].set_title('Rewards over Episodes')

  for ax in axs:
        ax.grid(True)
        
  # Adjust layout
  plt.tight_layout()

  # Show the plot
  plt.show()

# %%
def test(envs, q_net, max_steps):

  rewardsF = []
  n_fruit_eaten = []
  n_wall_hit = []
  n_body_hit = []
  fruit_data = []
  head_data = []
  body_data = []

  for _ in trange(1, max_steps + 1):

      # Update epsilon value
      epsilon = 0

      # Do one step in the environment
      _, _, rewards, _ = collect_one_step(envs, q_net, epsilon)

      # Save data for plot
      n_fruit_eaten.append(np.count_nonzero(np.array(rewards) == .5))
      n_wall_hit.append(np.count_nonzero(np.array(rewards) == -.1))
      n_body_hit.append(np.count_nonzero(np.array(rewards) == -.2))
      rewardsF.append(np.mean(np.array(rewards)))

      # Save the data of the game rendering
      heads = np.argwhere(envs.boards == envs.HEAD) # find the head in the board for all the games
      fruits = np.argwhere(envs.boards == envs.FRUIT) # find the fruit in the board for all the games
      bodies = np.argwhere(envs.boards == envs.BODY)
      filtered_body = [[row[1], row[2]] for row in bodies if row[0] == 0]

      fruit_data.append([fruits[0][1],fruits[0][2]])
      head_data.append([heads[0][1],heads[0][2]])
      body_data.append(filtered_body)

  print("Testing complete!")

  return rewardsF, n_fruit_eaten, n_wall_hit, n_body_hit, fruit_data, head_data, body_data


# %% [markdown]
# # Test

# %%
batch_size = 256

# Initialize the environment
envs = get_env(batch_size)
action_dim = 4  # Dimensionality of the action space

# Create an instance of the QN model
q_net_test = DDQN(action_dim)

q_net_test.model(envs.to_state())

q_net_test.model.load_weights('./weights/weights_ddqn.weights.h5')

max_steps = int(20000)

rewards_ts, n_fruit_eaten_ts, n_wall_hit_ts, n_body_hit_ts, fruit_data_ts, head_data_ts, body_data_ts = test(envs, q_net_test, max_steps)

# %%
#plotting(max_steps, n_fruit_eaten_ts, n_wall_hit_ts, n_body_hit_ts, rewards_ts)

rewards_ep = np.mean(np.reshape(rewards_ts, (100, 200)), axis=1)
n_fruit_eaten_ep = np.mean(np.reshape(n_fruit_eaten_ts, (100, 200)), axis=1)
n_wall_hit_ep = np.mean(np.reshape(n_wall_hit_ts, (100, 200)), axis=1)
n_body_hit_ep = np.mean(np.reshape(n_body_hit_ts, (100, 200)), axis=1)
max_steps = rewards_ep.shape[0]

plotting(max_steps, n_fruit_eaten_ep, n_wall_hit_ep, n_body_hit_ep, rewards_ep)

# %%
print("The average reward is:", np.mean(rewards_ts))

print("Finished")


