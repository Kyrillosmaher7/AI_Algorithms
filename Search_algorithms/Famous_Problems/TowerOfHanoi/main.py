
from typing import List, Tuple, Dict, Set
import networkx as nx
import matplotlib.pyplot as plt

from Search_algorithms.Famous_Problems.TowerOfHanoi.Hanoi import HanoiState, analyze_search_tree, bfs_hanoi, print_search_tree

def visualize_search_tree(search_tree: Dict):
    """Visualize the search tree using matplotlib."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        # Add nodes and edges
        for state_id, info in search_tree.items():
            G.add_node(state_id, 
                      label=str(info['state'])[:30] + "...",
                      depth=info['depth'])
            
            for child_id in info['children']:
                G.add_edge(state_id, child_id)
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(15, 10))
        
        # Draw nodes by depth
        depths = [info['depth'] for info in search_tree.values()]
        colors = plt.cm.viridis([d/max(depths) for d in depths])
        
        nx.draw_networkx_nodes(G, pos, node_color=colors, 
                              node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', 
                              arrowsize=10, alpha=0.5)
        
        # Add labels
        labels = {node: f"Depth {data['depth']}" 
                 for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Towers of Hanoi State-Space Search Tree (BFS)", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Install networkx and matplotlib for visualization:")
        print("pip install networkx matplotlib")


def generate_all_possible_states():
    """Generate all possible legal states for 3-disk Towers of Hanoi."""
    from itertools import product
    
    all_states = []
    disks = [1, 2, 3]
    
    # Generate all possible distributions of 3 disks among 3 rods
    for distribution in product([0, 1, 2], repeat=3):
        rods = [[], [], []]
        
        # Place each disk according to distribution
        for disk, rod_idx in zip([1, 2, 3], distribution):
            rods[rod_idx].append(disk)
        
        # Sort each rod (largest at bottom)
        for rod in rods:
            rod.sort(reverse=True)
        
        # Check if state is legal (no larger disk on top of smaller)
        legal = True
        for rod in rods:
            if rod != sorted(rod, reverse=True):
                legal = False
                break
        
        if legal:
            state = HanoiState(rods)
            all_states.append(state)
    
    return all_states


def main():
    """Main function to demonstrate BFS for Towers of Hanoi."""
    print("="*80)
    print("TOWERS OF HANOI PROBLEM - BFS STATE-SPACE SEARCH")
    print("="*80)
    print("Rules:")
    print("1. Only one disk may be moved at a time")
    print("2. Each move takes the top disk from one rod to another")
    print("3. No disk may be placed on top of a smaller disk")
    print("="*80)
    
    # Initial state: all disks on rod 1
    initial_rods = [[3, 2, 1], [], []]  # Rod 1 has disks 3(largest), 2, 1(smallest)
    initial_state = HanoiState(initial_rods)
    
    print("\nINITIAL STATE:")
    print(initial_state.display())
    
    print("\nRUNNING BREADTH-FIRST SEARCH...")
    solution_path, search_tree, visited = bfs_hanoi(initial_state)
    
    if solution_path:
        print("\n✓ SOLUTION FOUND!")
        print(f"Number of moves: {len(solution_path)}")
        print(f"Optimal moves (2^n - 1 for n=3): {2**3 - 1}")
        
        print("\nSOLUTION PATH:")
        # Simulate the solution
        current_state = initial_state
        print(f"Step 0: Initial state")
        print(current_state.display())
        
        for i, action in enumerate(solution_path, 1):
            # Find and apply the move
            moves = current_state.get_legal_moves()
            for from_rod, to_rod, action_desc in moves:
                if action_desc == action:
                    current_state = current_state.apply_move(from_rod, to_rod, action)
                    break
            
            print(f"\nStep {i}: {action}")
            print(current_state.display())
        
        print("\n" + "="*60)
        print(f"GOAL REACHED in {len(solution_path)} moves!")
        print("="*60)
    else:
        print("\n✗ NO SOLUTION FOUND!")
    
    # Print search tree
    print_search_tree(search_tree)
    
    # Analyze search
    analyze_search_tree(search_tree, visited)
    
    # Show all possible states
    print("\n" + "="*60)
    print("ALL POSSIBLE LEGAL STATES (27 total for 3 disks):")
    print("="*60)
    
    all_states = generate_all_possible_states()
    for i, state in enumerate(all_states, 1):
        print(f"State {i:2d}: {state}")
    
    print(f"\nTotal legal states: {len(all_states)}")
    
    # Try visualization (optional)
    try:
        visualize = input("\nVisualize search tree? (y/n): ").lower().strip()
        if visualize == 'y':
            visualize_search_tree(search_tree)
    except:
        pass


def test_different_starting_positions():
    """Test BFS with different starting positions."""
    print("\n" + "="*80)
    print("TESTING DIFFERENT STARTING POSITIONS")
    print("="*80)
    
    test_cases = [
        ("Standard", [[3, 2, 1], [], []]),
        ("Disk 1 moved", [[3, 2], [1], []]),
        ("Two disks moved", [[3], [2], [1]]),
        ("Almost solved", [[], [], [3, 2, 1]]),
    ]
    
    for name, rods in test_cases:
        print(f"\nTest: {name}")
        print(f"Initial configuration: {rods}")
        
        initial_state = HanoiState(rods)
        solution_path, search_tree, visited = bfs_hanoi(initial_state)
        
        if solution_path:
            print(f"  Solution found: {len(solution_path)} moves")
            if len(solution_path) <= 10:
                print(f"  Path: {' → '.join(solution_path)}")
        else:
            print(f"  No solution found")
        
        print(f"  States explored: {len(visited)}")

main()
test_different_starting_positions()