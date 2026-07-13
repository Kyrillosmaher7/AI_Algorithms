from typing import List, Dict, Optional, Tuple
import heapq
import random

class EightPuzzleState:
    """Class representing a state in the 8-Puzzle problem."""

    def __init__(self, board, empty_tile_pos, moves=0):
        """Initialize the state with the board configuration and empty tile position."""
        self.board = board  # 2D list representing the board
        self.empty_tile_pos = empty_tile_pos  # (row, col) of the empty tile
        self.moves = moves  # Number of moves taken to reach this state

    def is_goal(self):
        """
        Check if the current state is the goal state.
        The goal state is defined as:
            1 2 3
            4 5 6  
            7 8 0
        where 0 represents the empty tile.
        Returns True if the current state is the goal state, False otherwise.
        """
        goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        return self.board_flat() == goal  

    def board_flat(self):
        """Flatten the 2D board into a 1D list."""
        return [tile for row in self.board for tile in row]

    def get_possible_moves(self):
        """
        Get possible moves for the empty tile.
        Returns a list of (row, col) positions where the empty tile can move.
        """
        possible_moves = []
        row, col = self.empty_tile_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                possible_moves.append((new_row, new_col))
        return possible_moves

    def move(self, new_empty_tile_pos):
        """
        Move the empty tile to a new position.
        Returns a new EightPuzzleState with the updated board and empty tile position.
        """
        row, col = self.empty_tile_pos
        new_row, new_col = new_empty_tile_pos
        new_board = [r[:] for r in self.board]  # Deep copy of the board
        # Swap the empty tile with the adjacent tile
        new_board[row][col], new_board[new_row][new_col] = new_board[new_row][new_col], new_board[row][col]
        return EightPuzzleState(new_board, (new_row, new_col), self.moves + 1)

    def heuristic(self) -> int:
        """Calculate the Manhattan distance heuristic for the current state."""
        distance = 0
        for r in range(3):
            for c in range(3):
                tile = self.board[r][c]
                if tile != 0:  # Skip the empty tile
                    goal_r = (tile - 1) // 3
                    goal_c = (tile - 1) % 3
                    distance += abs(r - goal_r) + abs(c - goal_c)
        return distance

    @classmethod
    def from_list(cls, flat_list: List[int], moves: int = 0):
        """
        Create a state from a flat list (9 elements).
        """
        board = [flat_list[i*3:(i+1)*3] for i in range(3)]
        # Find empty tile position
        for r in range(3):
            for c in range(3):
                if board[r][c] == 0:
                    return cls(board, (r, c), moves)
        raise ValueError("No empty tile (0) found in the list")

    def is_solvable(self) -> bool:
        """
        Check if the puzzle state is solvable.
        A state is solvable if the number of inversions is even.
        """
        flat = self.board_flat()
        inversions = 0
        for i in range(len(flat)):
            for j in range(i+1, len(flat)):
                if flat[i] != 0 and flat[j] != 0 and flat[i] > flat[j]:
                    inversions += 1
        return inversions % 2 == 0

    def get_state_id(self) -> str:
        """
        Get a unique string identifier for the state.
        """
        return ''.join(str(tile) for tile in self.board_flat())

    def __str__(self):
        """String representation of the state for easy visualization."""
        return '\n'.join([' '.join([str(tile) if tile != 0 else ' ' for tile in row]) for row in self.board])

    def __lt__(self, other):
        """
        Less-than comparison for priority queue.
        Compare by f-score (heuristic + moves) for A* algorithm.
        """
        return (self.heuristic() + self.moves) < (other.heuristic() + other.moves)

    def __eq__(self, other):
        """Check if two states have the same board configuration."""
        if not isinstance(other, EightPuzzleState):
            return False
        return self.board_flat() == other.board_flat()

    def __hash__(self):
        """Hash based on the board configuration for use in sets/dictionaries."""
        return hash(tuple(self.board_flat()))

    @classmethod
    def random_state(cls):
        """
        Generate a random solvable 8-puzzle state.
        """
        tiles = list(range(9))
        random.shuffle(tiles)
        state = cls.from_list(tiles)
        
        # Check if solvable (inversions must be even)
        while not state.is_solvable():
            random.shuffle(tiles)
            state = cls.from_list(tiles)
        
        return state


def a_star_8puzzle(initial_state: EightPuzzleState) -> Tuple[Optional[List[str]], int, int]:
    """
    A* search specifically for 8-puzzle.
    
    Args:
        initial_state: Starting state
        
    Returns:
        Tuple of (solution_path, total_cost, nodes_expanded)
        Returns (None, 0, nodes_expanded) if no solution
    """
    # Check if already solved
    if initial_state.is_goal():
        return [], 0, 0
    
    # Frontier: (f_score, state)
    frontier = []
    heapq.heappush(frontier, (initial_state.heuristic(), initial_state))
    
    # Track visited states
    came_from = {initial_state: None}
    g_score = {initial_state: 0}  # Actual cost from start
    closed_set = set()
    nodes_expanded = 0
    
    while frontier:
        # Get state with lowest f_score
        f_current, current = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # Skip if we already found a better path to this state
        if current in closed_set and g_score[current] < f_current - current.heuristic():
            continue
            
        if current.is_goal():
            # Reconstruct solution path
            solution_path = []
            total_cost = g_score[current]
            
            # Backtrack to get moves
            while came_from[current] is not None:
                parent = came_from[current]
                # Find which move was taken
                for move_pos in parent.get_possible_moves():
                    new_state = parent.move(move_pos)
                    if new_state == current:
                        # Determine move direction
                        pr, pc = parent.empty_tile_pos
                        cr, cc = current.empty_tile_pos
                        if cr < pr:
                            solution_path.append("Up")
                        elif cr > pr:
                            solution_path.append("Down")
                        elif cc < pc:
                            solution_path.append("Left")
                        else:
                            solution_path.append("Right")
                        break
                current = parent
            
            solution_path.reverse()
            return solution_path, total_cost, nodes_expanded
        
        closed_set.add(current)
        
        # Generate successor states
        for move_pos in current.get_possible_moves():
            neighbor = current.move(move_pos)
            
            # Skip if already in closed set with better g_score
            if neighbor in closed_set:
                continue
                
            tentative_g = g_score[current] + 1  # Each move costs 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + neighbor.heuristic()
                heapq.heappush(frontier, (f_score, neighbor))
    
    return None, 0, nodes_expanded  # No solution found


def bfs_8puzzle(initial_state: EightPuzzleState) -> Tuple[Optional[List[str]], int, int]:
    """
    BFS search for 8-puzzle.
    
    Args:
        initial_state: Starting state
        
    Returns:
        Tuple of (solution_path, total_cost, nodes_expanded)
        Returns (None, 0, nodes_expanded) if no solution
    """
    from collections import deque
    
    if initial_state.is_goal():
        return [], 0, 0
    
    queue = deque([initial_state])
    visited = set()
    came_from = {initial_state: None}
    nodes_expanded = 0
    
    while queue:
        current = queue.popleft()
        nodes_expanded += 1
        visited.add(current)
        
        if current.is_goal():
            # Reconstruct path
            solution_path = []
            total_cost = current.moves
            
            while came_from[current] is not None:
                parent = came_from[current]
                # Find move direction
                for move_pos in parent.get_possible_moves():
                    new_state = parent.move(move_pos)
                    if new_state == current:
                        pr, pc = parent.empty_tile_pos
                        cr, cc = current.empty_tile_pos
                        if cr < pr:
                            solution_path.append("Up")
                        elif cr > pr:
                            solution_path.append("Down")
                        elif cc < pc:
                            solution_path.append("Left")
                        else:
                            solution_path.append("Right")
                        break
                current = parent
            
            solution_path.reverse()
            return solution_path, total_cost, nodes_expanded
        
        for move_pos in current.get_possible_moves():
            neighbor = current.move(move_pos)
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                queue.append(neighbor)
    
    return None, 0, nodes_expanded


def dfs_8puzzle(initial_state: EightPuzzleState, depth_limit: int = 50) -> Tuple[Optional[List[str]], int, int]:
    """
    Depth-Limited DFS search for 8-puzzle.
    
    Args:
        initial_state: Starting state
        depth_limit: Maximum depth to search
        
    Returns:
        Tuple of (solution_path, total_cost, nodes_expanded)
        Returns (None, 0, nodes_expanded) if no solution
    """
    stack = [(initial_state, [])]  # (state, path_so_far)
    visited = set()
    nodes_expanded = 0
    
    while stack:
        current, path = stack.pop()
        nodes_expanded += 1
        
        if current.is_goal():
            total_cost = current.moves
            return path, total_cost, nodes_expanded
        
        if len(path) >= depth_limit:
            continue
            
        if current in visited:
            continue
            
        visited.add(current)
        
        # Generate successors in reverse order for DFS behavior
        moves = current.get_possible_moves()
        for move_pos in reversed(moves):
            neighbor = current.move(move_pos)
            if neighbor not in visited:
                # Determine move direction
                pr, pc = current.empty_tile_pos
                cr, cc = neighbor.empty_tile_pos
                if cr < pr:
                    move_dir = "Up"
                elif cr > pr:
                    move_dir = "Down"
                elif cc < pc:
                    move_dir = "Left"
                else:
                    move_dir = "Right"
                
                stack.append((neighbor, path + [move_dir]))
    
    return None, 0, nodes_expanded


