# %% [markdown]
# Questa versione fa A2C

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
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(n_actions, activation='softmax')
        ])
        self.critic = tf.keras.Sequential([
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
state = env.to_state()
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
    states = envs.to_state()
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
    next_states = envs.to_state()

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

# %% [markdown]
# # Training Loop

# %%

def learn(a2c_net, optimizer_actor, optimizer_critic, gamma, states, next_states, actions, rewards):

  # Inside the training loop
  with tf.GradientTape(persistent=True) as tape_actor, tf.GradientTape(persistent=True) as tape_critic:
      # Compute the states value functions
      val = a2c_net.critic(states)
      #print(val)
      # Compute the next states value functions
      next_val = a2c_net.critic(next_states)
      #print(next_val)
      # Compute the actions probs
      action_probs = a2c_net.actor(states)
      #print(action_probs)
      # Compute the discounted rewards
      td_target = rewards + gamma * next_val # remember (1-dones)
      #print(td_target.shape)
      # Compute advantages
      advantages = (td_target - val)

      # Convert 'actions' to one-hot encoded tensor
      actions_one_hot = tf.one_hot(actions[:, 0], depth=action_probs.shape[-1])

      # Multiply 'action_probs' by 'actions_one_hot' to select the corresponding probabilities
      gathered_probs = tf.reduce_sum(action_probs * actions_one_hot, axis=1)

      # Compute the actor loss
      actor_loss = -tf.math.log(gathered_probs) * tf.squeeze(advantages)
      actor_loss = tf.reduce_mean(actor_loss)
      # Compute the critic loss
      critic_loss = tf.reduce_mean(tf.square(tf.squeeze(advantages)))

  # Compute the gradients from the loss
  grads_actor = tape_actor.gradient(actor_loss, a2c_net.actor.trainable_weights)
  grads_critic = tape_critic.gradient(critic_loss, a2c_net.critic.trainable_weights)

  # Apply the gradients to the model's parameters
  optimizer_actor.apply_gradients(zip(grads_actor, a2c_net.actor.trainable_weights))
  optimizer_critic.apply_gradients(zip(grads_critic, a2c_net.critic.trainable_weights))

  loss = actor_loss + critic_loss

  return loss

# %%
def train(envs, a2c_net, optimizer_actor, optimizer_critic, gamma, max_steps, TEST):

  rewardsF = []
  n_fruit_eaten = []
  n_wall_hit = []
  n_body_hit = []

  for step in trange(1, max_steps + 1):

      # Do one step in the environment
      states, actions, rewards, next_states = collect_one_step(envs, a2c_net,TEST)
      #print(actions[:, 0])
      n_fruit_eaten.append(np.count_nonzero(np.array(rewards) == .5))
      n_wall_hit.append(np.count_nonzero(np.array(rewards) == -.1))
      n_body_hit.append(np.count_nonzero(np.array(rewards) == -.2))

      mean_reward = np.mean(np.array(rewards))
      rewardsF.append(mean_reward)

      if step % 1000 == 0:
         print(f"Mean Reward: {round(mean_reward, 4)}")

      # Learning
      loss = learn(a2c_net, optimizer_actor, optimizer_critic, gamma, states, next_states, actions, rewards)
      # print(loss)

  print("Training complete!")

  return rewardsF, n_fruit_eaten, n_wall_hit, n_body_hit

# %% [markdown]
# # Do it

# %%
# Initialize the environment
envs = get_env(256)
action_dim = 4  # Dimensionality of the action space

# Create an instance of the QN model
a2c_net_train = A2C(action_dim)
# Create optimizer for actor and critic separately
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=3e-4)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Compile the model
a2c_net_train.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()],
                 metrics=[['accuracy'], ['accuracy']])

gamma = 0.99

max_steps = int(20_000)

TEST = False

iterations_per_second = 12

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
rewards_tr, n_fruit_eaten_tr, n_wall_hit_tr, n_body_hit_tr = train(envs, a2c_net_train, optimizer_actor, optimizer_critic, gamma, max_steps, TEST)

# %% [markdown]
# ### Save the weights

# %%
a2c_net_train.save_weights('weights_a2c.weights.h5')

# %% [markdown]
# ### Plot the results

# %%
# plotting(max_steps, n_fruit_eaten_tr, n_wall_hit_tr, n_body_hit_tr, rewards_tr)

# %%
rewards_ep = np.mean(np.reshape(rewards_tr, (100, 200)), axis=1)
n_fruit_eaten_ep = np.mean(np.reshape(n_fruit_eaten_tr, (100, 200)), axis=1)
n_wall_hit_ep = np.mean(np.reshape(n_wall_hit_tr, (100, 200)), axis=1)
n_body_hit_ep = np.mean(np.reshape(n_body_hit_tr, (100, 200)), axis=1)
max_steps = rewards_ep.shape[0]

plotting(max_steps, n_fruit_eaten_ep, n_wall_hit_ep, n_body_hit_ep, rewards_ep)

# %% [markdown]
# ### Save the data

# %%
# Save the data
# np.savez("./data/train/data_a2c.npz", max_steps=max_steps, n_fruit_eaten_ep=n_fruit_eaten_ep, n_wall_hit_ep=n_wall_hit_ep, n_body_hit_ep=n_body_hit_ep, rewards_ep=rewards_ep)