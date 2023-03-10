# Monte Carlo Tree Search (MCTS) for Go
This repository contains a Python implementation of the Monte Carlo Tree Search (MCTS) algorithm for playing the game of Go. The MCTS algorithm is a heuristic search algorithm that uses random simulations to find the best move in a game.

## Overview
The game of Go is played on a 15x15 board. The objective of the game is to place five stones in a row, either horizontally, vertically, or diagonally. The game is played by two players, Black and White, who take turns placing stones on the board.

The MCTS algorithm works by building a search tree from the current game state. At each node in the tree, the algorithm chooses the move that has the highest Upper Confidence Bound (UCB) score, which balances the exploitation of the best moves found so far with the exploration of new moves. The algorithm then simulates a game from the chosen move to the end of the game, using a policy network to guide the simulations. The result of the simulation is then backpropagated up the tree, updating the UCB scores of the nodes visited during the search.

The policy network used in the simulations is a convolutional neural network (CNN) that takes as input the current game state and outputs a probability distribution over the possible moves.

## Dependencies
The implementation requires the following Python packages:

- numpy
- tensorflow
- keras

## Usage
```
git clone https://github.com/jackmazac/gomcts.git
pip install -r requirements.txt
python3 mcts.py
```

### Notes
To play against the AI, run the human_play() function in the mcts.py file. You can choose to play against the AI or watch the AI play against itself.

To train the policy network, run the train() function in the train.py file. You can customize the hyperparameters and the number of epochs to train the network for.

## Credits
The implementation is based on the following resources:

AlphaGo Zero: Mastering the Game of Go without Human Knowledge (https://www.nature.com/articles/nature24270)
A Simple Alpha(Go) Zero Tutorial (https://web.stanford.edu/~surag/posts/alphazero.html)


## Technical Breakdown
This is a Python code for the implementation of a game-playing algorithm called Monte Carlo Tree Search (MCTS). The code defines three classes: Player, Node, and Game.

###### Player
The Player class represents a player in the game. Each player has a color (black or white) and a flag indicating whether they have passed their turn.

###### Node
The Node class represents a node in the search tree used by the MCTS algorithm. Each node has a move, a parent node, a list of child nodes, a visit count, a score, a game state, a player, and a list of untried moves. The select_child method selects the child node with the highest Upper Confidence Bound (UCB) score, which balances between exploitation of the current best move and exploration of other moves. The add_child method adds a child node to the current node, and the update method updates the visit count and score of the current node.

###### Game
The Game class represents the game itself, and it has several methods that define the rules of the game, such as get_legal_moves and get_winner. The main function of the class is play_game, which takes two players as input and plays the game until there is a winner. The MCTS algorithm is used to select moves for the players.

The code also imports several modules, including math, random, sys, multiprocessing, and numpy, and it uses the TensorFlow and Keras libraries for deep learning.

###### MCTSPlayer
This Python code defines two player classes for the game. The MCTSPlayer class implements a player that uses the Monte Carlo Tree Search algorithm to select moves. The DQNPlayer class implements a player that uses a deep neural network to estimate the Q-values of each possible move and chooses the move with the highest Q-value.

The MCTSPlayer class has several parameters, including the number of simulations to run, the exploration weight for the UCB score, a rollout policy (which is a function that plays out a game from a given state), the number of processes to use, and two parameters for progressive widening (a technique used to balance exploration and exploitation in the search). The get_move method uses the MCTS algorithm to select a move for the player.

###### DQNPlayer
The DQNPlayer class takes a pre-trained model as input and uses it to estimate the Q-values for each possible move. The get_move method converts the game board to an input tensor for the model, gets the predicted Q-values for each legal move, and chooses the move with the highest Q-value.

The get_model function defines the architecture of the neural network used by the DQNPlayer class. It is a convolutional neural network with several convolutional layers, batch normalization layers, and activation functions. The final layer is a softmax activation that outputs the Q-values for each possible move.

###### Board
The Board class defines the game board and its rules, while the Player class defines the behavior of a player. The MCTSPlayer class implements a player that uses Monte Carlo Tree Search to select moves, while the DQNPlayer class uses a Deep Q-Network to select moves.

###### DQNAgent
The DQNAgent class defines the Deep Q-Network used by the DQNPlayer. It consists of a convolutional neural network that takes the current game state as input and outputs Q-values for each possible move. The train method trains the network using a batch of transitions sampled from a replay buffer.

The Node class is used by the MCTSPlayer to represent nodes in the search tree. It keeps track of the state of the game, the move that was made to get to this state, the player who made the move, and the statistics of the node (number of visits, total score, and children).

###### Human Play
The human_play() function implements a game where a human player can play against an AI player that uses the Monte Carlo tree search algorithm to select its moves. The reward() function computes the reward given a game state and a player, where a reward of 1 is given if the player wins, -1 if the opponent wins, and a small fraction of the difference in the number of stones between the two players if the game ends in a tie. 

###### GameState
Finally, the GameState class defines the state of a game of a board game where two players take turns placing stones on a board of a certain size and the goal is to create a sequence of five stones in a row either horizontally, vertically, or diagonally. The get_state() method of this class returns the state of the game in a format suitable for use as input to a neural network.

This code implements a deep Q-learning agent that learns to play the game of Gomoku. The agent is trained for a specified number of episodes using a replay buffer to store transitions, and the agent's policy is updated using a Q-learning algorithm.

The code begins by defining the size of the game board, the starting value of epsilon (the probability of taking a random action), and the parameters for epsilon decay. It then initializes the game state and creates an instance of the ReplayBuffer class to store transitions.

Next, the code creates an instance of the DQNAgent class, which is defined elsewhere and implements the deep Q-learning algorithm. The agent's initial policy is used to select an action for the first state of the game. The code then defines the number of episodes to train the agent and creates two players to play the game.

The main training loop iterates over the specified number of episodes. In each episode, the game state and player are initialized, and the game is played until it ends. At each step of the game, the agent selects an action using its current policy, the action is executed in the game, and the agent observes the next state and the reward for the current transition. The transition is added to the replay buffer, and the agent is trained on a random batch of transitions from the buffer. The value of epsilon is updated, and the player's turn is switched. When the game ends, the results of the episode are printed.

###### Conclusion
Overall, this code implements a simple and straightforward deep Q-learning algorithm to learn to play the game of Gomoku. However, there are several ways in which the algorithm could be improved, such as using a more sophisticated neural network architecture or implementing a prioritized replay buffer.


## Known Errors
```
  File "mcts.py", line 577, in <module>
    loss = agent.train(buffer, batch_size, gamma)
  File "mcts.py", line 154, in train
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
  File "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/keras/optimizers/optimizer_experimental/optimizer.py", line 1139, in apply_gradients
    grads_and_vars = self.aggregate_gradients(grads_and_vars)
  File "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/keras/optimizers/optimizer_experimental/optimizer.py", line 1105, in aggregate_gradients
    return optimizer_utils.all_reduce_sum_gradients(grads_and_vars)
  File "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/keras/optimizers/optimizer_v2/utils.py", line 33, in all_reduce_sum_gradients
    filtered_grads_and_vars = filter_empty_gradients(grads_and_vars)
  File "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/keras/optimizers/optimizer_v2/utils.py", line 77, in filter_empty_gradients
    raise ValueError(
ValueError: No gradients provided for any variable: (['conv2d/kernel:0', 'conv2d/bias:0', 'dense/kernel:0', 'dense/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0'],). Provided `grads_and_vars` is ((None, <tf.Variable 'conv2d/kernel:0' shape=(3, 3, 2, 32) dtype=float32, numpy=
array([[[[-0.08934976,  0.01405928,  0.02807374, -0.01024981,
           0.06801187, -0.04710032, -0.11098963,  0.0198378 ,
          -0.04272735,  0.05877328, -0.13044961,  0.04954763,
           0.03278144, -0.02765062, -0.00408141, -0.11108641,
          -0.00857355, -0.03227676, -0.13001761, -0.05738963,
           0.10237843, -0.01187775,  0.00021173, -0.00080292,
          -0.01091342, -0.00741163, -0.02901498, -0.07763106,
          -0.05344309,  0.04362081, -0.00196178, -0.0540084 ],
         [ 0.08540709, -0.05074079, -0.0308057 , -0.0203925 ,
          -0.01393639, -0.10498659, -0.0817897 , -0.07782716,
          -0.0443462 , -0.12250549, -0.10285464,  0.12954907,
           0.12129174, -0.05491611,  0.02806287, -0.13048823,
           0.05374396,  0.11342146,  0.03383619, -0.05357229,
           0.09354311, -0.00311598, -0.02007834, -0.11623272,
          -0.01071808, -0.08485019, -0.03112012, -0.07160435,
           0.04760934, -0.048501  , -0.11841919,  0.06527287]],

        [[ 0.12300442, -0.13914838,  0.04617365, -0.03908375,
          -0.12923695,  0.00888623, -0.07047813,  0.02551092,
           0.06932938, -0.10664344, -0.06874683,  0.09673321,
          -0.05787882,  0.02634062, -0.07033157, -0.00475246,
           0.07276829,  0.12334804,  0.04540084,  0.06927742,
          -0.04591024,  0.05892316,  0.00547001, -0.01713112,
           0.02752265, -0.10673985, -0.09918707,  0.01586071,
           0.00538509,  0.09867531, -0.00119352,  0.02527583],
         [-0.07605243,  0.00389242, -0.07873815, -0.11665492,
          -0.04423673, -0.0602847 ,  0.08115424,  0.03973883,
           0.05591343, -0.13753808,  0.11886187,  0.11211278,
          -0.03223079, -0.0557022 ,  0.12960757,  0.08079538,
           0.00890559,  0.05326211, -0.08789336, -0.00836104,
           0.05827962,  0.12601714, -0.12938616,  0.06284642,
          -0.0402104 ,  0.02944024,  0.1166283 ,  0.13171525,
          -0.08232623, -0.12535101,  0.09374617,  0.13993506]],

        [[ 0.00583641, -0.0603283 ,  0.01561847,  0.05722487,
           0.13203476,  0.1226161 ,  0.098095  ,  0.01369281,
          -0.06803566, -0.03273824,  0.00668237, -0.05373695,
          -0.00757205,  0.13213815, -0.0386418 , -0.05924144,
          -0.1294138 , -0.05840668, -0.06832591,  0.0570502 ,
           0.0062992 , -0.07488522,  0.12022342,  0.03260086,
          -0.0959332 , -0.08820447, -0.0133414 ,  0.03119338,
           0.12281977,  0.04050139,  0.10414174, -0.01988   ],
         [-0.09383865,  0.01313947, -0.01406819,  0.00791042,
           0.1173069 ,  0.04071929,  0.06672622, -0.00223   ,
           0.02988407,  0.13866656,  0.01444511,  0.12899746,
          -0.13459486, -0.05303141,  0.02014007,  0.02963491,
          -0.02138444, -0.07693297,  0.08843049, -0.10675023,
          -0.06718341,  0.10014105,  0.04186377,  0.13814898,
          -0.02425634,  0.03510636, -0.01689082,  0.00621054,
           0.06638306, -0.12920882,  0.10844414,  0.13623269]]],


       [[[-0.09747837, -0.13350576,  0.05569196,  0.12428947,
           0.00920068, -0.06273318,  0.00540234, -0.09485166,
           0.01672138,  0.08921681, -0.03720599, -0.13347825,
          -0.07172674,  0.04453704, -0.05506331,  0.10979864,
           0.12344055,  0.03585155,  0.09798747,  0.06260578,
           0.10344787,  0.03001423,  0.10069211,  0.10821889,
           0.04307088,  0.09661263, -0.1173344 , -0.1241408 ,
          -0.05925623, -0.0579917 , -0.06748771, -0.12124899],
         [ 0.12808432,  0.04434837, -0.08016153, -0.10144185,
          -0.05811222,  0.0062737 ,  0.06672646,  0.06399207,
          -0.1206733 , -0.0987004 ,  0.04589605, -0.07583466,
          -0.00437009, -0.08666141, -0.04896706,  0.0055424 ,
           0.03929678, -0.08695039, -0.13040724, -0.05346606,
          -0.03655258,  0.10574517, -0.09309602,  0.10815589,
          -0.02980053,  0.07657665,  0.10954018,  0.01386641,
           0.02161169, -0.04645766, -0.13656269, -0.07673927]],

        [[-0.0925238 , -0.06080491,  0.14000852,  0.0784059 ,
           0.10702497, -0.10437701,  0.0899964 ,  0.02913213,
          -0.12795639,  0.13819449,  0.00830072,  0.10265346,
           0.0300177 , -0.12824959,  0.02392095,  0.01112798,
          -0.09874889,  0.11350946, -0.0291995 , -0.13534914,
           0.10724367,  0.00527881, -0.09461533,  0.11515664,
           0.03277433, -0.02489197,  0.09268211,  0.02000345,
           0.05476367,  0.08118175, -0.13164294, -0.0197949 ],
         [-0.05226021, -0.04679228,  0.13508157,  0.04687974,
           0.135711  ,  0.08343419, -0.00918299, -0.12559116,
           0.05092764, -0.02933338,  0.0344694 ,  0.09538355,
          -0.0981826 ,  0.0604983 ,  0.00908791,  0.01072696,
           0.12540473,  0.09047107,  0.00455332,  0.08011402,
          -0.05537096,  0.02949175,  0.0762603 ,  0.02679349,
           0.12535037,  0.10249291,  0.11538284, -0.07464054,
           0.08800741, -0.04990759,  0.12621139, -0.04657444]],

        [[-0.00611652, -0.09509744, -0.1313164 ,  0.00483583,
           0.00230792,  0.09625323,  0.1054879 , -0.08570816,
           0.02507745, -0.01642305, -0.08469228,  0.04032604,
           0.04040363,  0.04471564, -0.11365378, -0.11048054,
          -0.06972326, -0.02353509,  0.04317755, -0.08503228,
           0.03779858, -0.06228331,  0.01292074,  0.07352784,
           0.11933737, -0.07560043,  0.12699594,  0.06630674,
          -0.10031499,  0.12240519,  0.08472693,  0.03322199],
         [-0.08938771, -0.1160211 , -0.13838118,  0.02595736,
          -0.06637879, -0.0914521 ,  0.0181845 ,  0.02633141,
          -0.0629152 , -0.1279967 ,  0.09565488,  0.13902105,
          -0.0635606 , -0.05077514,  0.11122794,  0.13722132,
          -0.02264731, -0.10280409, -0.13420138,  0.06999736,
          -0.05689359, -0.11677984, -0.07610809, -0.00624308,
          -0.09781069,  0.03210463, -0.12285479,  0.06752954,
          -0.03316641,  0.05932891,  0.03302132, -0.07800888]]],


       [[[ 0.00695552, -0.12847337, -0.02137459,  0.10434844,
           0.0286579 , -0.01680114, -0.13778019,  0.12500767,
          -0.11851338, -0.13097978, -0.00845879,  0.07076702,
          -0.04547749,  0.05025944,  0.0613382 , -0.11883407,
          -0.09329332,  0.00214821, -0.04320419, -0.05867559,
           0.01462498,  0.0865332 ,  0.1004111 , -0.00464259,
          -0.08133443,  0.08816956, -0.09277229,  0.06080498,
           0.08666448,  0.0980507 ,  0.00559273,  0.04880674],
         [ 0.03832327,  0.045137  ,  0.0596139 , -0.08833925,
          -0.01314528,  0.04858045, -0.01573718, -0.06131524,
           0.02752309, -0.10445604, -0.07432258, -0.00708029,
           0.07586037,  0.0801039 ,  0.13932206,  0.02570349,
           0.04363279, -0.11992858,  0.06008753,  0.09291308,
           0.10490453, -0.00056548, -0.09429225, -0.10517806,
           0.08433048, -0.13057952,  0.11642383,  0.01465644,
           0.03874272,  0.03058572, -0.08408744,  0.07931121]],

        [[-0.03968552,  0.12475751,  0.09498747,  0.12637274,
          -0.12915225,  0.1197661 ,  0.0646158 ,  0.0234355 ,
           0.12422259,  0.08192912,  0.07972488,  0.11087234,
           0.1012    ,  0.12294935, -0.13565227,  0.02715194,
           0.12520044,  0.10454306,  0.095911  , -0.03780289,
           0.02010912,  0.08627331,  0.0006227 ,  0.12014844,
          -0.04100928, -0.12925315,  0.11650859, -0.03575096,
          -0.08934379,  0.13814394, -0.03698946,  0.10508497],
         [-0.02033387,  0.0485023 , -0.10140879,  0.02286194,
          -0.04759599, -0.03636588, -0.04789148, -0.0363328 ,
           0.10818881,  0.08594736, -0.08822227, -0.11539455,
           0.02685609, -0.04977225,  0.0187635 ,  0.00886457,
          -0.03408731, -0.11984023, -0.11169553, -0.04104059,
           0.08449256,  0.02424276,  0.12879919, -0.11898568,
          -0.10084064,  0.05081473, -0.05537619, -0.13272762,
           0.00260071,  0.04438856, -0.100069  , -0.03056613]],

        [[-0.11789631,  0.13695373,  0.05155993, -0.06987906,
           0.13496004,  0.11442949, -0.03898613,  0.00654772,
          -0.10161795, -0.08368668,  0.13738383,  0.13023861,
          -0.06661438,  0.05876814, -0.04979905, -0.06177288,
          -0.06303241,  0.02301851, -0.01239465, -0.05052518,
           0.08337533, -0.08702658, -0.03208376,  0.00234659,
           0.11494504, -0.0412583 ,  0.07510462,  0.00725442,
          -0.10455238,  0.03136541, -0.07001551, -0.08546975],
         [ 0.02570753, -0.04731546,  0.00039294,  0.03834033,
           0.07701357, -0.00590558,  0.05610797, -0.11517364,
           0.01503477, -0.05931927, -0.02720086,  0.0276344 ,
           0.07696125, -0.12471696,  0.07966746, -0.03357708,
           0.03906158, -0.0820914 ,  0.0870399 ,  0.01149371,
           0.02230488,  0.06464328, -0.08273013,  0.02296333,
           0.13683061,  0.06267285, -0.05260649, -0.05122914,
           0.03173979, -0.06886011, -0.10549605,  0.02892648]]]],
      dtype=float32)>), (None, <tf.Variable 'conv2d/bias:0' shape=(32,) dtype=float32, numpy=
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      dtype=float32)>), (None, <tf.Variable 'dense/kernel:0' shape=(1568, 64) dtype=float32, numpy=
array([[ 0.01507834,  0.05545085, -0.00454035, ...,  0.03052551,
         0.0421054 , -0.04046758],
       [ 0.01131348,  0.05480502, -0.05821431, ..., -0.02430096,
        -0.04138787, -0.04377305],
       [-0.00783302, -0.03812552, -0.03268608, ...,  0.04912656,
         0.00587445, -0.04140004],
       ...,
       [-0.04938833, -0.02255026,  0.03202795, ...,  0.02228383,
        -0.03383069,  0.01305357],
       [ 0.04366335,  0.03854635, -0.01881556, ..., -0.02596293,
        -0.00125437, -0.00127   ],
       [-0.01984697, -0.0552798 ,  0.02199176, ..., -0.03297583,
         0.0063177 , -0.02614494]], dtype=float32)>), (None, <tf.Variable 'dense/bias:0' shape=(64,) dtype=float32, numpy=
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>), (None, <tf.Variable 'dense_1/kernel:0' shape=(64, 81) dtype=float32, numpy=
array([[ 0.17155339, -0.07659817, -0.09625082, ..., -0.04120974,
         0.11738931, -0.08390559],
       [-0.12504257, -0.1299319 ,  0.02277561, ...,  0.03511237,
        -0.05374868,  0.02876706],
       [-0.02798167,  0.07901068, -0.1360476 , ...,  0.04575467,
         0.04500303, -0.01352435],
       ...,
       [-0.15316412,  0.1029924 , -0.01839441, ..., -0.07598703,
        -0.10277553,  0.0899079 ],
       [-0.15387021,  0.05561231, -0.07244948, ..., -0.0450234 ,
         0.18849511, -0.1711509 ],
       [-0.01915541,  0.00173442,  0.19271131, ..., -0.08740007,
         0.20052962,  0.11578681]], dtype=float32)>), (None, <tf.Variable 'dense_1/bias:0' shape=(81,) dtype=float32, numpy=
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>)).
```