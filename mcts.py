import math
import random
import sys
from multiprocessing import Pool
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np

sys.setrecursionlimit(100000)


def print_board(board):
    for row in board:
        print(' '.join(str(cell) for cell in row))


class Player:
    def __init__(self, color):
        self.color = color
        self.passed = False

    def opponent(self):
        return BLACK if self.color == WHITE else WHITE

    def __str__(self):
        return self.color


BLACK = Player(color='B')
WHITE = Player(color='W')


class Node:
    def __init__(self, move=None, parent=None, state=None, player=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
        self.state = state
        self.player = player
        self.untried_moves = state.get_legal_moves(player)

    def select_child(self, exploration_weight):
        total_visits = sum(child.visits for child in self.children)
        log_total_visits = math.log(total_visits)

        def ucb_score(child):
            exploitation = child.score / child.visits
            exploration = exploration_weight * math.sqrt(log_total_visits / child.visits)
            return exploitation + exploration

        return max(self.children, key=ucb_score)

    def add_child(self, move, state, player):
        child = Node(move=move, parent=self, state=state, player=player)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def update(self, score):
        self.visits += 1
        self.score += score

    def __repr__(self):
        return f"Node(move={self.move}, visits={self.visits}, score={self.score})"

class DQNAgent:
    def __init__(self, state_shape, num_actions, optimizer):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.action_space = np.arange(self.num_actions)
        self.model = self.build_model()
        self.optimizer = optimizer

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(self.state_shape[0], self.state_shape[1], self.state_shape[2])))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.state_shape[0] * self.state_shape[1], activation='linear'))
        return model


    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            # choose a random action
            return np.random.choice(self.num_actions)
        else:
            # choose the action with the highest Q-value
            state_flat = []
            for s in state:
                if isinstance(s, np.ndarray) and s.ndim == 2:
                    # If the element is an array with 2 dimensions, add a singleton dimension to make it 3D
                    s = s[np.newaxis, ...]
                state_flat.append(s)
            # Filter out any zero-dimensional arrays in state_flat
            state_flat = [s for s in state_flat if s.size > 0]
            if len(state_flat) == 0:
                # if there are no valid state tensors, return a random action
                return np.random.choice(self.num_actions)
            state_tensor = np.concatenate(state_flat, axis=-1)
            state_tensor = np.reshape(state_tensor, (1, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
            q_values = self.model(state_tensor)
            best_action = np.argmax(q_values)
            return best_action




    def train(self, replay_buffer, batch_size, gamma):
        # Sample a batch of transitions from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        
        # Convert the state and next state batches to tensors
        state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        
        # Compute the Q-values for the current state and next state using the model
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)
        
        # Compute the target Q-values using the Bellman equation
        target_q_values = reward_batch + gamma * tf.reduce_max(next_q_values, axis=1) * (1 - done_batch)
        
        # Use the action batch to index into the Q-values and get the Q-value for each action taken
        # during the transition
        action_mask = tf.one_hot(action_batch, self.num_actions)
        q_values = tf.reduce_sum(q_values * action_mask, axis=1)
        
        # Compute the loss between the Q-values and the target Q-values
        loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        # Compute the gradients of the loss with respect to the model parameters
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply the gradients to the model using an optimizer
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()


class MCTSPlayer(Player):
    def __init__(self, num_simulations, exploration_weight, rollout_policy=None, num_processes=1, pw_c=1/2, pw_alpha=1/4, color=None):
        super().__init__(color)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.states = set()
        self.rollout_policy = rollout_policy or self.random_rollout
        self.num_processes = num_processes
        self.pool = None
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha
        self.dqn_agent = DQNAgent(state_shape=(B, B, 2), num_actions=B*B)

    def get_move(self, board):
        root = Node(state=board, player=board.current_player)
        for i in range(self.num_simulations):
            node = root
            state = board.copy()

            # Select
            while node.untried_moves == [] and node.children != []:
                node = node.select_child(self.exploration_weight)
                state.play_move(node.move, node.player)

            # Progressive Widening
            if node.untried_moves != []:
                pw_n = len(node.children) ** self.pw_alpha
                pw_t = len(state.get_legal_moves(node.player)) ** self.pw_c
                if pw_n < pw_t:
                    move = random.choice(node.untried_moves)
                    state.play_move(move, node.player)
                    node = node.add_child(move, state, node.player.opponent())

            # Expand
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                state.play_move(move, node.player)
                node = node.add_child(move, state, node.player.opponent())

            # Simulate
            score = self.rollout_policy(state)

            # Backpropagate
            while node is not None:
                node.update(score)
                node = node.parent

        legal_moves = board.get_legal_moves(board.current_player)
        if not legal_moves:
            return None
        elif len(root.children) == 0:
            return random.choice(legal_moves)
        else:
            return max(root.children, key=lambda c: c.visits).move


class DQNPlayer(Player):
    def __init__(self, model, color):
        super().__init__(color)
        self.model = model

    def get_move(self, board):
        legal_moves = board.get_legal_moves(self)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        # convert board to input tensor for model
        board_array = np.array(board.grid)
        board_array[board_array == self.opponent().color] = 'O'
        board_array[board_array == self.color] = 'X'
        board_array[board_array == '.'] = 'E'
        board_array = np.expand_dims(board_array, axis=0)

        # get predicted Q values for each legal move
        q_values = self.model.predict(board_array)[0]
        legal_move_indices = [board.size * move[0] + move[1] for move in legal_moves]
        legal_q_values = q_values[legal_move_indices]

        # choose move with highest Q value
        max_q_value = max(legal_q_values)
        best_moves = [move for i, move in enumerate(legal_moves) if legal_q_values[i] == max_q_value]
        return random.choice(best_moves)

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(B, B, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(B * B),
        tf.keras.layers.Activation('softmax')
    ])
    return model



class Board:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))
        self.current_player = BLACK
        self.last_move = None

    def play(self, move, player):
        x, y = move
        self.grid[x][y] = 1 if player.color == BLACK.color else -1
        self.current_player = player.opponent()
        self.last_move = move

    def get_state(self):
        return self.grid.copy()

    def get_legal_moves(self, player):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x][y] == '.' and self.is_valid_move(x, y, player):
                    moves.append((x, y))
        return moves

    def is_valid_move(self, x, y, player):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.grid[x][y] != '.':
            return False
        if self.is_self_capture(x, y, player):
            return False
        if self.is_ko(x, y, player):
            return False
        return True

    def is_self_capture(self, x, y, player):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x+dx, y+dy
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                continue
            if self.grid[nx][ny] == player.opponent().color:
                liberties = self.get_liberties_iterative(nx, ny)
                if len(liberties) == 1 and (x, y) in liberties:
                    return True
        return False

    def is_ko(self, x, y, player):
        if self.last_move is None:
            return False
        if (x, y) != self.last_move:
            return False
        new_board = self.copy()
        new_board.play((x, y), player)
        if new_board.grid == self.grid:
            return True
        return False
    
    def get_liberties_iterative(self, x, y, visited=None):
        if visited is None:
            visited = set()
        liberties = set()
        queue = deque([(x, y)])
        visited.add((x, y))

        while queue:
            x, y = queue.popleft()
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                if self.grid[nx][ny] == '.':
                    liberties.add((nx, ny))
                elif self.grid[nx][ny] == self.grid[x][y]:
                    queue.append((nx, ny))

        return liberties

    def get_neighbors(self, x, y):
        """
        Returns a list of (x, y) coordinates of the neighbors of the given cell.
        """
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < self.size - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < self.size - 1:
            neighbors.append((x, y + 1))
        return neighbors


    def is_captured_iterative(self, x, y, visited=None):
        visited = visited or set()
        player = self.grid[x][y]
        queue = [(x, y)]
        while queue:
            x, y = queue.pop()
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                    continue
                if self.grid[nx][ny] == '.':
                    return False
                if self.grid[nx][ny] == player:
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
                else:
                    # opponent stone
                    liberties = self.get_liberties_iterative(nx, ny)
                    if len(liberties) > 1:
                        return False
        return True

    def copy(self):
        # Create a new board object with the same size as the current board
        new_board = Board(self.size)
        # Copy the grid of the current board to the new board
        new_board.grid = [row[:] for row in self.grid]
        # Set the current player and last move of the new board to the current player and last move of the current board
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        # Return the new board
        return new_board

    def get_score(self, player):
        score = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x][y] == player.color:
                    score += 1
        return score
    
    def is_game_over(self):
        # The game is over if both players pass in a row or the board is full
        return self.current_player.passed and self.current_player.opponent().passed or self.is_board_full()

    def is_board_full(self):
        # The board is full if there are no empty spaces
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x][y] == '.':
                    return False
        return True

    def do_move(self, move, player):
        x, y = move
        self.grid[x][y] = str(player)

    def get_winner(self):
        empty_points = self.get_empty_points()
        if not empty_points:
            black_score = sum(len(group) for group in self.groups[BLACK])
            white_score = sum(len(group) for group in self.groups[WHITE])
            if black_score > white_score:
                return BLACK
            elif white_score > black_score:
                return WHITE
            else:
                return None
        else:
            return None
        
    def get_empty_points(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i][j] == '.']



def human_play():
    B = 6
    board = Board(B)
    print_board(board.grid)

    BLACK = Player('B')
    WHITE = Player('W')
    players = [BLACK, WHITE]
    turn = 0
    mcts_player = MCTSPlayer(num_simulations=10, exploration_weight=1.4)

    while True:
        #mode = input("Enter '1' to play against the AI or '2' to watch the AI play against itself: ")
        mode = '2'
        if mode == '1':
            human_player = WHITE
            mcts_player = MCTSPlayer(num_simulations=10, exploration_weight=1.4, color=BLACK)
            break
        elif mode == '2':
            human_player = None
            mcts_player_1 = MCTSPlayer(num_simulations=10, exploration_weight=1.4, color=BLACK)
            mcts_player_2 = MCTSPlayer(num_simulations=10, exploration_weight=1.4, color=WHITE)
            break
        else:
            print("Invalid mode. Try again.")

    times_passed = 0
    while True:
        if human_player is None:
            print("MCTS is thinking...")
            if turn % 2 == 0:
                move = mcts_player_1.get_move(board)
            else:
                move = mcts_player_2.get_move(board)
            player = players[turn % 2]
            if move is not None:
                print(f"{player.color} placed a stone at ({move[0]}, {move[1]})")
                board.play(move, player)
                print_board(board.grid)
            else:
                print(f"{player.color} passed")
                times_passed += 1
        else:
            player = players[turn % 2]
            if player == human_player:
                x, y = map(int, input(f"{player.color} player's turn. Enter move (row, col): ").split())
                move = (x, y)
                while not board.is_valid_move(x, y, player):
                    print("Invalid move. Try again.")
                    x, y = map(int, input(f"{player.color} player's turn. Enter move (row, col): ").split())
                    move = (x, y)
                board.play(move, player)
                print(f"{player.color} placed a stone at ({move[0]}, {move[1]})")
                print_board(board.grid)
            else:
                print("MCTS is thinking...")
                move = mcts_player.get_move(board)
                if move is not None:
                    print(f"{player.color} placed a stone at ({move[0]}, {move[1]})")
                    board.play(move, player)
                    print_board(board.grid)
                else:
                    print(f"{player.color} passed")
                    times_passed += 1
        if times_passed == 3:
            winner = board.get_winner()
            if winner is None:
                print("Game ended in a tie.")
            else:
                print(f"{winner} wins!")
            break
        turn += 1

        if board.is_game_over():
            winner = board.get_winner()
            if winner is None:
                print("Game ended in a tie.")
            else:
                print(f"{winner} wins!")
            break

        if board.is_board_full():
            print("Board is full. Game ended in a tie.")
            break
    
def reward(game, player):
    # check if the game has ended
    result = game.get_winner()
    if result == player:
        # AI player wins
        return 1
    elif result != 0 and result != player:
        # opponent wins
        return -1
    else:
        # game is still ongoing
        # compute the difference between the number of stones of the AI player and its opponent
        ai_stones = np.count_nonzero(game.board == player.color)
        opp_stones = np.count_nonzero(game.board == player.opponent().color)
        stone_diff = ai_stones - opp_stones
        
        # compute the reward based on the stone difference
        return 0.01 * stone_diff



# define the game state
class GameState:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.player = 1
        self.action_to_move = {i*self.board_size+j: (i, j) for i in range(self.board_size) for j in range(self.board_size)}
    
    def get_legal_moves(self):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def make_move(self, action):
        # Get the indices corresponding to the chosen action
        move = self.action_to_move[action]
        if move is not None and isinstance(move, (list, tuple)) and len(move) == 2:
            i, j = move
            # Make the move
            self.board[i][j] = self.player
            self.player = self.player % 2 + 1
        else:
            print(f"Invalid move: {move}")

    
    def get_winner(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    continue
                if self.check_sequence(i, j, 1, 0):
                    return self.board[i][j]
                if self.check_sequence(i, j, 0, 1):
                    return self.board[i][j]
                if self.check_sequence(i, j, 1, 1):
                    return self.board[i][j]
                if self.check_sequence(i, j, 1, -1):
                    return self.board[i][j]
        return 0
    
    def check_sequence(self, i, j, di, dj):
        for k in range(5):
            if i-di*k < 0 or i-di*k >= self.board_size or j-dj*k < 0 or j-dj*k >= self.board_size or self.board[i][j] != self.board[i-di*k][j-dj*k]:
                return False
        return True
    
    def get_state(self):
        board = self.board.copy()
        board[self.board == 3 - self.player] = 2  # replace opponent's stones with 2
        board[self.board == self.player] = 1  # replace player's stones with 1
        state = np.zeros((self.board_size, self.board_size, 2))
        state[:, :, 0] = (board == 1)
        state[:, :, 1] = (board == 2)
        return state


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_transition(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch)
    
board_size = 9
epsilon = 1.0  # starting value for epsilon
epsilon_min = 0.1  # minimum value for epsilon
epsilon_decay = 0.99  # decay rate for epsilon
game_state = GameState(board_size)
state = game_state.get_state()
# create a new instance of the ReplayBuffer class
buffer = ReplayBuffer(max_size=10000)
batch_size = 32 # or any other value you want
gamma = 0.99
optimizer = Adam(learning_rate=0.001)
agent = DQNAgent(state_shape=(board_size, board_size, 2), num_actions=board_size * board_size, optimizer=optimizer)
action = agent.act(state, epsilon)
num_episodes = 10  # or any other value you want
# create two players
player1 = Player("black")
player2 = Player("white")

# train the agent for a specified number of episodes
for episode in range(num_episodes):
    # initialize the game state and the player
    if episode % 2 == 0:
        game_state = GameState(board_size)
        player = player1
    else:
        game_state = GameState(board_size)
        player = player2
    done = False
    
    # play the game until it ends
    while not done:
        # get the current state of the game
        state = game_state.get_state()
        
        # get an action from the agent
        action = agent.act(state, epsilon)
        
        # execute the action in the game
        game_state.make_move(action)
        
        # get the new state of the game
        next_state = game_state.get_state()
        
        # get the reward for the current transition
        reward = reward(game_state, player)
        
        # add the transition to the replay buffer
        buffer.add_transition(state, action, reward, next_state, done)
        
        # train the agent using the replay buffer
        loss = agent.train(buffer, batch_size, gamma)
        
        # update the value of epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        # switch the player's turn
        player = player1 if player == player2 else player2
        
        # check if the game has ended
        done = game_state.check_game_over()
    
    # print the results of the episode
    print(f"Episode {episode}: Loss = {loss}, Epsilon = {epsilon}")
