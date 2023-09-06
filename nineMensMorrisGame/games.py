"""Games or Adversarial Search (Chapter 5)"""

import copy
import itertools
import random
import time
from collections import namedtuple

import numpy as np

GameState = namedtuple('GameState', 'to_move, utility, board, moves')
StochasticGameState = namedtuple('StochasticGameState', 'to_move, utility, board, moves, chance')


def gen_state(to_move='X', x_positions=[], o_positions=[], h=6, v=6):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""
    unavailable_spots = [       (0,1), (0,2),      (0,4), (0,5),
                         (1,0),        (1,2),      (1,4),        (1,6),
                         (2,0), (2,1),                    (2,5), (2,6),
                                             (3,3),
                         (4,0), (4,1),                    (4,5), (4,6),
                         (5,0),        (5,2),      (5,4),        (5,6),
                                (6,1), (6,2),      (6,4), (6,5)
                         ]

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(unavailable_spots) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)


# ______________________________________________________________________________
# MinMax Search


# GameState = namedtuple('GameState', 'to_move, utility, board, moves')
# x, y = minmax_decision(gen_state(to_move=player, x_positions=self.player1.poses, o_positions=self.player2.poses, h=6, v=6), self.game)
def minmax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)
    start_time = time.time()

    def max_value(state):
        if game.terminal_test(state) or time.time() - start_time > 5:
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state) or time.time() - start_time > 5:
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


# ______________________________________________________________________________


def expect_minmax(state, game):
    """
    [Figure 5.11]
    Return the best move for a player after dice are thrown. The game tree
	includes chance nodes along with min and max nodes.
	"""
    player = game.to_move(state)
    start_time = time.time()

    def max_value(state):
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(state, a))
        return v

    def min_value(state):
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a))
        return v

    def chance_node(state, action):
        res_state = game.result(state, action)
        if game.terminal_test(res_state) or time.time() - start_time > 5:
            return game.utility(res_state, player)
        sum_chances = 0
        num_chances = len(game.chances(res_state))
        for chance in game.chances(res_state):
            res_state = game.outcome(res_state, chance)
            util = 0
            if res_state.to_move == player:
                util = max_value(res_state)
            else:
                util = min_value(res_state)
            sum_chances += util * game.probability(chance)
        return sum_chances / num_chances

    # Body of expect_minmax:
    return max(game.actions(state), key=lambda a: chance_node(state, a), default=None)


def alpha_beta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)
    start_time = time.time()

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state) or time.time() - start_time > 5:
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state) or time.time() - start_time > 5:
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)
    start_time = time.time()

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        if depth == -1 and time.time() - start_time > 5:
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)

        if depth == -1 and time.time() - start_time > 5:
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

def second_phase(state, game, search_type, possible_moves, depth=-1, eval_fn=None):
    # to do
    time.sleep(5) #check how slow game will actually take
    if search_type == 'expectiminimax':
        pass
    elif search_type == 'alpha_beta':
        pass
    elif search_type == 'alpha_beta_cutoff':
        pass
    elif search_type == 'minmax':
        pass


# ______________________________________________________________________________
# Players for Games

def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    return alpha_beta_search(state, game)


def minmax_player(game,state):
    return minmax_decision(state,game)


def expect_minmax_player(game, state):
    return expect_minmax(state, game)


# ______________________________________________________________________________
# Some Sample Games


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class NMensMorris(Game):
    '''
           A simple Gameboard for 9MensMorris game class. This class contains all game specifics logic like
           what next move to take, if a point (completing a row or diagonal or column) been achieved, if
           game is terminated, and so on. For deciding on next move, there are 3 phases to the game:
            •	Placing pieces on vacant points (9 turns each)
            •	Moving placed pieces to adjacent points.
            •	Moving pieces to any vacant point (when the player has been reduced to 3 men)

            This class governs all the logic of the game. This means it has to check validity of each player's
            move, as well as deciding on next move for the AI player. This class receives the game state,
            in form of the list of rows of cells on the board.

        '''
    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        board = []  # an array of 7 rows, each row an array of element from set {'X', 'O', '-'}.
                    #. 'X' means occupied by Human player, 'O' is occupied by AI, '-' means still vacant
        self.initial = GameState(to_move='X', utility=0, board={}, moves={})
        self.direction = {'X': -1, 'O': 1}

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def is_legal_move(self, board, start, end, player):
        """ can be used to check if a move from start to end positions by player. This function can
        be called for example by get_all_moves() for checking validity of on-board piece moves
        """
        pass

    def chances(self, state):
        """Return a list of all possible dice rolls at a state."""
        dice_rolls = list(itertools.combinations_with_replacement([1, 2, 3, 4, 5, 6], 2))
        return dice_rolls

    def outcome(self, state, chance):
        """Return the state which is the outcome of a dice roll."""
        dice = tuple(map((self.direction[state.to_move]).__mul__, chance))
        return StochasticGameState(to_move=state.to_move,
                                   utility=state.utility,
                                   board=state.board,
                                   moves=state.moves, chance=dice)

    def probability(self, chance):
        """Return the probability of occurrence of a dice roll."""
        return 1 / 36 if chance[0] == chance[1] else 1 / 18

    def get_all_moves(self, board, player):
        """All possible moves for a player. Depending of the state of the game, it can
        include all positions to put a new piece, or all position to move the current pieces.
        The design and format is for students' to do"""
        pass

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.game
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k
