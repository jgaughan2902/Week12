import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def update_board(current_board):
    '''
    Update each cell's dead or alive status on the board.

    Parameters:
    current_board (np.array): A binary NumPy array used to
    execute a step in Conway's game of life.

    Return value:
    updated_board (np.array): An array of integers representing
    the binary form of the next board to be used.
    '''

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    neighbors = convolve2d(current_board, kernel, 
                           mode = 'same', boundary = 'fill', fillvalue = 0)


    # Apply the rules of the game by defining when a cell survives (which is
    # when a cell is alive and has 2 or 3 surrounding alive cells) or is 
    # born (which is when the cell is dead but there are 3 alive neighbor cells).
    survive = (current_board == 1) & ((neighbors == 2) | (neighbors == 3))
    born = (current_board == 0) & (neighbors == 3)

    # The updated board consists of alive cells (defined as 'survive' or
    # 'born' cells) and marks them as alive while every other cell is
    # considered dead.
    updated_board = (survive | born).astype(int)

    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)