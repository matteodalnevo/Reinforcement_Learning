# %% [markdown]
# DDQN with epsilon greedy decay
# 
# NN: Flatted input > Dense(64) > Dense(64) > Dense(4)

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
# # Plot function

# %%
def plotting(max_steps, n_fruit_eaten, n_wall_hit, n_body_hit, rewards):
  # Create a figure and axis objects
  fig, axs = plt.subplots(1, 4, figsize=(15, 5))

  # Plot n_fruit_eaten
  axs[0].plot(range(1, max_steps + 1), n_fruit_eaten)
  axs[0].set_xlabel('Training Steps')
  axs[0].set_ylabel('Eaten fruits')
  axs[0].set_title('Eaten fruits over Training Steps')

  # Plot n_wall_hit
  axs[1].plot(range(1, max_steps + 1), n_wall_hit)
  axs[1].set_xlabel('Training Steps')
  axs[1].set_ylabel('Wall hit')
  axs[1].set_title('Wall hit over Training Steps')

  # Plot n_body_hit
  axs[2].plot(range(1, max_steps + 1), n_body_hit)
  axs[2].set_xlabel('Training Steps')
  axs[2].set_ylabel('Body hit')
  axs[2].set_title('Body hit over Training Steps')

  # Plot n_fruit_eaten
  axs[3].plot(range(1, max_steps + 1), rewards)
  axs[3].set_xlabel('Training Steps')
  axs[3].set_ylabel('Rewards')
  axs[3].set_title('Rewards over Training Steps')

  for ax in axs:
        ax.grid(True)
        
  # Adjust layout
  plt.tight_layout()

  # Show the plot
  plt.show()

# %% [markdown]
# ## Linear Schedule

# %%
def linear_schedule(initial_value: float, final_value: float, current_step: int, max_steps: int) -> float:
    """
    Linear schedule for the exploration rate (epsilon).
    Note: we clip the value so the schedule is constant after reaching the final value
    at `max_steps`.

    :param initial_value: Initial value of the schedule.
    :param final_value: Final value of the schedule.
    :param current_step: Current step of the schedule.
    :param max_steps: Maximum number of steps of the schedule.
    :return: The current value of the schedule.
    """
    # Compute current progress (in [0, 1], 0 being the start)
    progress = current_step / max_steps
    # Clip the progress so the schedule is constant after reaching the final value
    progress = min(progress, 1.0)
    current_value = initial_value + progress * (final_value - initial_value)

    return current_value

# %% [markdown]
# # Training Loop

# %%
def learn(q_net, target_q_net, optimizer, gamma, states, next_states, actions, rewards, batch_size):

  with tf.GradientTape(persistent=True) as tape:
    # Compute the Q-values for the next observations
    next_q_values = target_q_net.model(next_states)

    # Follow greedy policy: use the one with the highest value
    next_q_values = tf.reduce_max(next_q_values, axis=1)

    # TD error
    td_target = rewards + gamma * next_q_values[:, tf.newaxis] # * tf.cast(~dones, tf.float32)

    # Get current Q-values estimates for the replay_data
    q_values = q_net.model(states)

    # Use tf.gather to gather the action values corresponding to chosen action indices
    current_q_values = tf.gather(q_values, actions, batch_dims=1)

    # Compute the Mean Squared Error (MSE) loss
    loss = tf.reduce_mean(tf.square(current_q_values - td_target))

  # Compute gradients
  gradients = tape.gradient(loss, q_net.model.trainable_weights)
  # Apply gradients
  optimizer.apply_gradients(zip(gradients, q_net.model.trainable_weights))

  return loss

def update_target(q_net, target_q_net):
    """
    Update the weights of the target Q-network to match the main Q-network.

    :param q_net: The main Q-network
    :param target_q_net: The target Q-network
    """
    target_q_net.model.set_weights(q_net.model.get_weights())

# %%
def train(envs, q_net, target_q_net, optimizer, gamma, batch_size, max_steps, epsilon_start, epsilon_end, limit, update_target_frequency):

  rewardsF = []
  n_fruit_eaten = []
  n_wall_hit = []
  n_body_hit = []

  for step in trange(1, max_steps + 1):

      # Update epsilon value
      epsilon = linear_schedule(epsilon_start, epsilon_end, step, max_steps * limit)

      # Do one step in the environment
      states, actions, rewards, next_states = collect_one_step(envs, q_net, epsilon)

      n_fruit_eaten.append(np.count_nonzero(np.array(rewards) == .5))
      n_wall_hit.append(np.count_nonzero(np.array(rewards) == -.1))
      n_body_hit.append(np.count_nonzero(np.array(rewards) == -.2))
      rewardsF.append(np.mean(np.array(rewards)))

      if step % 1000 == 0:
         print(f"Mean Reward: {round(np.mean(np.array(rewards)), 4)} - Epsilon: {round(epsilon, 2)}")

      # Learning
      loss = learn(q_net, target_q_net, optimizer, gamma, states, next_states, actions, rewards, batch_size)

      # Periodically update the target network
      if step % update_target_frequency == 0:
          update_target(q_net, target_q_net)

  print("Training complete!")

  return rewardsF, n_fruit_eaten, n_wall_hit, n_body_hit

# %% [markdown]
# # Do it

# %%
batch_size = 256

# Initialize the environment
envs = get_env(batch_size)
action_dim = 4  # Dimensionality of the action space

# Create an instance of the QN model
q_net = DDQN(action_dim)
# Compile the model
q_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss=[tf.keras.losses.MeanSquaredError()],
                 metrics=[['accuracy']])

target_q_net = DDQN(action_dim)
# Compile the model
target_q_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss=[tf.keras.losses.MeanSquaredError()],
                 metrics=[['accuracy']])

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)

# Hyperparams
gamma = 0.99
max_steps = int(20_000)
epsilon_start = 1.0
epsilon_end = 0.1
limit = 0.1
update_target_frequency = 1000

iterations_per_second = 15

# Calculate total time in seconds
total_seconds = max_steps / iterations_per_second

# Convert seconds to days, hours, and minutes
total_days = total_seconds // (24 * 3600)
remaining_seconds = total_seconds % (24 * 3600)
total_hours = remaining_seconds // 3600
remaining_seconds %= 3600
total_minutes = remaining_seconds // 60

# Print the result
print(f"Training time: {int(total_days)} days, {int(total_hours)} hours, {int(total_minutes)} minutes.")

# %%
# Run training
rewards_tr, n_fruit_eaten_tr, n_wall_hit_tr, n_body_hit_tr = train(envs, q_net, target_q_net, optimizer, gamma, batch_size, max_steps, epsilon_start, epsilon_end, limit, update_target_frequency)

# %% [markdown]
# ### Save model

# %%
q_net.model.save_weights('weights_ddqn.weights.h5')

# %% [markdown]
# ### Plot

# %%
#plotting(max_steps, n_fruit_eaten_tr, n_wall_hit_tr, n_body_hit_tr, rewards_tr)

rewards_ep = np.mean(np.reshape(rewards_tr, (100, 200)), axis=1)
n_fruit_eaten_ep = np.mean(np.reshape(n_fruit_eaten_tr, (100, 200)), axis=1)
n_wall_hit_ep = np.mean(np.reshape(n_wall_hit_tr, (100, 200)), axis=1)
n_body_hit_ep = np.mean(np.reshape(n_body_hit_tr, (100, 200)), axis=1)
max_steps = rewards_ep.shape[0]

plotting(max_steps, n_fruit_eaten_ep, n_wall_hit_ep, n_body_hit_ep, rewards_ep)

# %%
# Save the data
np.savez("data_ddqn.npz", max_steps=max_steps, n_fruit_eaten_ep=n_fruit_eaten_ep, n_wall_hit_ep=n_wall_hit_ep, n_body_hit_ep=n_body_hit_ep, rewards_ep=rewards_ep)