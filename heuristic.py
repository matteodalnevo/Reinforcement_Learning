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

# %%
def is_colliding_with_body(x, y, body_positions):
    """
    Check if a position (x, y) would result in a collision with the snake's body.

    Parameters:
        x (int): The x-coordinate of the position to check.
        y (int): The y-coordinate of the position to check.
        body_positions (list): List of tuples representing the positions of the snake's body parts.

    Returns:
        bool: True if there is a collision, False otherwise.
    """
    if not body_positions:
        return False
    # Check if the given position (x, y) is in the list of body positions
    return [x, y] in body_positions

# %%
def baseline(env, actions, iter):
  """
    Determine the next action for the Snake agent based on the current game state.

    Parameters:
        env (object): The environment object representing the Snake game environment.
        actions (list): A list to store the actions chosen by the baseline Snake agent.
        iter (int): An integer representing the selected board in the game.

    Returns:
        list: Updated list of actions chosen by the Snake agent.
"""

  # Get the head position from the enviroment
  heads = np.argwhere(env.boards == env.HEAD)
  head_x, head_y = heads[iter][1], heads[iter][2]
  # Get the fruit position from the enviroment
  fruits = np.argwhere(env.boards == env.FRUIT)
  fruit_x, fruit_y = fruits[iter][1], fruits[iter][2]
  # Get the body pos
  bodies = np.argwhere(env.boards == env.BODY) # find the body parts in the board for all the games
  filtered_body = [[row[1], row[2]] for row in bodies if row[0] == 0]

  # Calculate distance to food in each direction
  dx, dy = fruit_x - head_x, fruit_y - head_y
  horizontal_distance = abs(dy)
  vertical_distance = abs(dx)

  if horizontal_distance < vertical_distance:
        # If food is to the left, move left
        if dy < 0 and not is_colliding_with_body(head_x, head_y - 1, filtered_body):
            actions.append(3)  # LEFT
        # If food is to the right, move right
        elif dy > 0 and not is_colliding_with_body(head_x, head_y + 1, filtered_body):
            actions.append(1)  # RIGHT
        # If food is directly above or below, prioritize vertical movement
        elif not is_colliding_with_body(head_x - 1, head_y, filtered_body) or not is_colliding_with_body(head_x + 1, head_y, filtered_body):
            # If food is above, move up
            if dx < 0:
                actions.append(2)  # UP
            # If food is below, move down
            elif dx > 0:
                actions.append(0)  # DOWN
        else:
            # Fallback: choose a random direction
            actions.append(random.choice([0, 1, 2, 3]))
  else:
        # If food is above, move up
        if dx < 0 and not is_colliding_with_body(head_x - 1, head_y, filtered_body):
            actions.append(2)  # UP
        # If food is below, move down
        elif dx > 0 and not is_colliding_with_body(head_x + 1, head_y, filtered_body):
            actions.append(0)  # DOWN
        # If food is directly to the left or right, prioritize horizontal movement
        elif not is_colliding_with_body(head_x, head_y - 1, filtered_body) or not is_colliding_with_body(head_x, head_y + 1, filtered_body):
            # If food is to the left, move left
            if dy < 0:
                actions.append(3)  # LEFT
            # If food is to the right, move right
            elif dy > 0:
                actions.append(1)  # RIGHT
        else:
            # Fallback: choose a random direction
            actions.append(random.choice([0, 1, 2, 3]))
  return actions

# %%
# Initialize variables
n_boards = 256
env = get_env(n_boards)
random_rewards = []
iter = 0
max_iterations = 20000
iteration_count = 0
boards = []
head_data = []
body_data = []
fruit_data = []
n_fruit_eaten = []
n_wall_hit = []
n_body_hit = []

###################################################################################################################
# Start the loop
print("START")
while iteration_count < max_iterations:
  if iteration_count % 1000 == 0:
    print("Iterations:", iteration_count)
  
  available = np.argwhere(env.boards[0] == env.EMPTY)
  if len(available) == 0:
    break  # Exit the loop if there are no available empty spaces

  actions = []
  for iter in range(n_boards): # compute the n_boards actions

    ################################################################################################################
    # Smart Baseline actions computation
    actions = baseline(env, actions, iter)

  ###################################################################################################################
  # Save the data of the first board
  heads = np.argwhere(env.boards == env.HEAD) # find the head in the board for all the games
  fruits = np.argwhere(env.boards == env.FRUIT) # find the fruit in the board for all the games
  bodies = np.argwhere(env.boards == env.BODY) # find the body parts in the board for all the games
  filtered_body = [[row[1], row[2]] for row in bodies if row[0] == 0]

  fruit_data.append([fruits[0][1],fruits[0][2]])
  head_data.append([heads[0][1],heads[0][2]])
  body_data.append(filtered_body)

  # compute rewards and final rewards
  tensor = tf.convert_to_tensor(actions)
  tensor = tf.expand_dims(tensor, axis=1)
  rewards = env.move(tensor)
  random_rewards.append(np.mean(rewards))

  # Save the metrics
  n_fruit_eaten.append(np.count_nonzero(np.array(rewards) == .5))
  n_wall_hit.append(np.count_nonzero(np.array(rewards) == -.1))
  n_body_hit.append(np.count_nonzero(np.array(rewards) == -.2))

  iteration_count += 1
###################################################################################################################

rewards_ep = np.mean(np.reshape(random_rewards, (100, 200)), axis=1)
# Plot the reward
plt.plot(range(len(rewards_ep)),rewards_ep)
plt.xlabel('Iter')
plt.ylabel('Mean Rewards')
plt.title('Plot of Rewards')
plt.grid(True)
plt.show()

# %%
print("The mean reward is:", np.mean(random_rewards))
print("The mean fruit eaten is:", np.mean(n_fruit_eaten))
print("The mean walls hit is:", np.mean(n_wall_hit))
print("The mean body hit is:", np.mean(np.mean(n_body_hit)))

# %%
# Save the data
#np.savez("data_baseline.npz", max_steps=100, n_fruit_eaten_ep=np.mean(n_fruit_eaten), n_wall_hit_ep=np.mean(n_wall_hit), n_body_hit_ep=np.mean(n_body_hit), rewards_ep=np.mean(random_rewards))