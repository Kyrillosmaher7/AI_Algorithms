"""
8-Puzzle search problem demo.

Run from the project root using:

    python -m Search_Problems.Famous_Problems.EightPuzzleProblem.main
"""
from Search_algorithms.Famous_Problems.EightPuzzleProblem.EightPuzzle import EightPuzzleState, a_star_8puzzle, bfs_8puzzle


    # Create initial state
initial_board = [
        [4, 6, 1],
        [8, 0, 7],
        [5, 2, 3]
]
initial_state = EightPuzzleState(initial_board, (1, 1))
    
print("Initial state:")
print(initial_state)
print()
    
# Test solvability
print(f"Is solvable? {initial_state.is_solvable()}")
print()
    
# Solve using A*
print("Solving with A*...")
solution, cost, nodes =a_star_8puzzle(initial_state)
if solution:
        print(f"Solution found in {len(solution)} moves:")
        print(" -> ".join(solution))
        print(f"Total cost: {cost}")
        print(f"Nodes expanded: {nodes}")
else:
        print("No solution found")
        print()
    
    # Solve using BFS
print("Solving with BFS...")
solution, cost, nodes = bfs_8puzzle(initial_state)
if solution:
        print(f"Solution found in {len(solution)} moves:")
        print(" -> ".join(solution))
        print(f"Total cost: {cost}")
        print(f"Nodes expanded: {nodes}")
else:
        print("No solution found")
        print()
    
# Generate random state
print("Generating New Random solvable state For Another Try Next Time....")
random_state = EightPuzzleState.random_state()
print(random_state)