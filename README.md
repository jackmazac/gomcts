# Monte Carlo Tree Search (MCTS) for Gomoku
This repository contains a Python implementation of the Monte Carlo Tree Search (MCTS) algorithm for playing the game of Gomoku. The MCTS algorithm is a heuristic search algorithm that uses random simulations to find the best move in a game.

## Overview
The game of Gomoku is played on a 15x15 board. The objective of the game is to place five stones in a row, either horizontally, vertically, or diagonally. The game is played by two players, Black and White, who take turns placing stones on the board.

The MCTS algorithm works by building a search tree from the current game state. At each node in the tree, the algorithm chooses the move that has the highest Upper Confidence Bound (UCB) score, which balances the exploitation of the best moves found so far with the exploration of new moves. The algorithm then simulates a game from the chosen move to the end of the game, using a policy network to guide the simulations. The result of the simulation is then backpropagated up the tree, updating the UCB scores of the nodes visited during the search.

The policy network used in the simulations is a convolutional neural network (CNN) that takes as input the current game state and outputs a probability distribution over the possible moves.

## Dependencies
The implementation requires the following Python packages:

numpy
tensorflow
keras
Usage
To play against the AI, run the human_play() function in the mcts.py file. You can choose to play against the AI or watch the AI play against itself.

To train the policy network, run the train() function in the train.py file. You can customize the hyperparameters and the number of epochs to train the network for.

## Credits
The implementation is based on the following resources:

AlphaGo Zero: Mastering the Game of Go without Human Knowledge (https://www.nature.com/articles/nature24270)
A Simple Alpha(Go) Zero Tutorial (https://web.stanford.edu/~surag/posts/alphazero.html)


## Technical Breakdown
This is a Python code for the implementation of a game-playing algorithm called Monte Carlo Tree Search (MCTS). The code defines three classes: Player, Node, and Game.

## Player
The Player class represents a player in the game. Each player has a color (black or white) and a flag indicating whether they have passed their turn.

## Node
The Node class represents a node in the search tree used by the MCTS algorithm. Each node has a move, a parent node, a list of child nodes, a visit count, a score, a game state, a player, and a list of untried moves. The select_child method selects the child node with the highest Upper Confidence Bound (UCB) score, which balances between exploitation of the current best move and exploration of other moves. The add_child method adds a child node to the current node, and the update method updates the visit count and score of the current node.

## Game
The Game class represents the game itself, and it has several methods that define the rules of the game, such as get_legal_moves and get_winner. The main function of the class is play_game, which takes two players as input and plays the game until there is a winner. The MCTS algorithm is used to select moves for the players.

The code also imports several modules, including math, random, sys, multiprocessing, and numpy, and it uses the TensorFlow and Keras libraries for deep learning.

## MCTSPlayer
This Python code defines two player classes for the game. The MCTSPlayer class implements a player that uses the Monte Carlo Tree Search algorithm to select moves. The DQNPlayer class implements a player that uses a deep neural network to estimate the Q-values of each possible move and chooses the move with the highest Q-value.

The MCTSPlayer class has several parameters, including the number of simulations to run, the exploration weight for the UCB score, a rollout policy (which is a function that plays out a game from a given state), the number of processes to use, and two parameters for progressive widening (a technique used to balance exploration and exploitation in the search). The get_move method uses the MCTS algorithm to select a move for the player.

## DQNPlayer
The DQNPlayer class takes a pre-trained model as input and uses it to estimate the Q-values for each possible move. The get_move method converts the game board to an input tensor for the model, gets the predicted Q-values for each legal move, and chooses the move with the highest Q-value.

The get_model function defines the architecture of the neural network used by the DQNPlayer class. It is a convolutional neural network with several convolutional layers, batch normalization layers, and activation functions. The final layer is a softmax activation that outputs the Q-values for each possible move.

## Board
The Board class defines the game board and its rules, while the Player class defines the behavior of a player. The MCTSPlayer class implements a player that uses Monte Carlo Tree Search to select moves, while the DQNPlayer class uses a Deep Q-Network to select moves.

## DQNAgent
The DQNAgent class defines the Deep Q-Network used by the DQNPlayer. It consists of a convolutional neural network that takes the current game state as input and outputs Q-values for each possible move. The train method trains the network using a batch of transitions sampled from a replay buffer.

The Node class is used by the MCTSPlayer to represent nodes in the search tree. It keeps track of the state of the game, the move that was made to get to this state, the player who made the move, and the statistics of the node (number of visits, total score, and children).

## Human Play
The human_play() function implements a game where a human player can play against an AI player that uses the Monte Carlo tree search algorithm to select its moves. The reward() function computes the reward given a game state and a player, where a reward of 1 is given if the player wins, -1 if the opponent wins, and a small fraction of the difference in the number of stones between the two players if the game ends in a tie. 

## GameState
Finally, the GameState class defines the state of a game of a board game where two players take turns placing stones on a board of a certain size and the goal is to create a sequence of five stones in a row either horizontally, vertically, or diagonally. The get_state() method of this class returns the state of the game in a format suitable for use as input to a neural network.

This code implements a deep Q-learning agent that learns to play the game of Gomoku. The agent is trained for a specified number of episodes using a replay buffer to store transitions, and the agent's policy is updated using a Q-learning algorithm.

The code begins by defining the size of the game board, the starting value of epsilon (the probability of taking a random action), and the parameters for epsilon decay. It then initializes the game state and creates an instance of the ReplayBuffer class to store transitions.

Next, the code creates an instance of the DQNAgent class, which is defined elsewhere and implements the deep Q-learning algorithm. The agent's initial policy is used to select an action for the first state of the game. The code then defines the number of episodes to train the agent and creates two players to play the game.

The main training loop iterates over the specified number of episodes. In each episode, the game state and player are initialized, and the game is played until it ends. At each step of the game, the agent selects an action using its current policy, the action is executed in the game, and the agent observes the next state and the reward for the current transition. The transition is added to the replay buffer, and the agent is trained on a random batch of transitions from the buffer. The value of epsilon is updated, and the player's turn is switched. When the game ends, the results of the episode are printed.

## Conclusion
Overall, this code implements a simple and straightforward deep Q-learning algorithm to learn to play the game of Gomoku. However, there are several ways in which the algorithm could be improved, such as using a more sophisticated neural network architecture or implementing a prioritized replay buffer.

