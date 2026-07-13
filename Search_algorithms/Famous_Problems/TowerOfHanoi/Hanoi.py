
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple


class HanoiState:
    """Class representing a state in the Towers of Hanoi problem with 3 disks."""
    
    def __init__(self, rods: List[List[int]], depth: int = 0, parent=None, action: str = None):
        """
        Initialize the state.
        
        Args:
            rods: List of 3 rods, each is a list of disks (3=largest, 1=smallest)
            depth: Depth in search tree
            parent: Parent state
            action: Action taken to reach this state
        """
        self.rods = [rod.copy() for rod in rods]  # Deep copy
        self.depth = depth
        self.parent = parent
        self.action = action
        
    def is_goal(self, target_rod: int = 2) -> bool:
        """Check if all disks are on the target rod in correct order."""
        # Rods 0 and 1 should be empty
        if len(self.rods[0]) > 0 or len(self.rods[1]) > 0:
            return False
        
        # Rod 2 should have all disks in correct order [3, 2, 1]
        return self.rods[2] == [3, 2, 1]
    
    def get_legal_moves(self) -> List[Tuple[int, int, str]]:
        """Get all legal moves from current state.
        Returns list of (from_rod, to_rod, action_description)"""
        moves = []
        
        for from_rod in range(3):
            if not self.rods[from_rod]:  # Rod is empty
                continue
                
            # Top disk on from_rod
            disk = self.rods[from_rod][-1]
            
            for to_rod in range(3):
                if from_rod == to_rod:
                    continue
                    
                # Check if move is legal
                if not self.rods[to_rod] or self.rods[to_rod][-1] > disk:
                    # Create action description
                    action = f"Move disk {disk} from rod {from_rod+1} to rod {to_rod+1}"
                    moves.append((from_rod, to_rod, action))
        
        return moves
    
    def apply_move(self, from_rod: int, to_rod: int, action: str) -> 'HanoiState':
        """Apply move and return new state."""
        new_rods = [rod.copy() for rod in self.rods]
        disk = new_rods[from_rod].pop()
        new_rods[to_rod].append(disk)
        
        return HanoiState(new_rods, self.depth + 1, self, action)
    
    def get_state_id(self) -> str:
        """Get unique string identifier for state."""
        return str(self.rods)
    
    def __str__(self) -> str:
        """String representation of state."""
        rods_str = []
        for i, rod in enumerate(self.rods):
            rods_str.append(f"Rod {i+1}: {rod}")
        return " | ".join(rods_str)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HanoiState):
            return False
        return self.rods == other.rods
    
    def __hash__(self) -> int:
        return hash(str(self.rods))
    
    def display(self) -> str:
        """Display the towers visually."""
        max_height = 3
        output = []
        
        for level in range(max_height - 1, -1, -1):
            row = []
            for rod in self.rods:
                if level < len(rod):
                    disk = rod[level]
                    row.append(f" {'=' * disk}{' ' * (3-disk)}")
                else:
                    row.append("    |   ")
            output.append(" ".join(row))
        
        output.append("-" * 30)
        output.append("  Rod 1    Rod 2    Rod 3")
        return "\n".join(output)


def bfs_hanoi(initial_state: HanoiState) -> Tuple[List[str], Dict, Set]:
    """
    Breadth-First Search for Towers of Hanoi.
    
    Args:
        initial_state: Starting state
        
    Returns:
        Tuple of (solution_path, search_tree, visited_states)
    """
    if initial_state.is_goal():
        return [], {}, {initial_state}
    
    # Queue for BFS
    queue = deque([initial_state])
    
    # Track visited states and build search tree
    visited = {initial_state}
    search_tree = {}
    search_tree[initial_state.get_state_id()] = {
        'state': initial_state,
        'parent': None,
        'children': [],
        'depth': 0
    }
    
    solution_path = []
    solution_found = False
    
    while queue and not solution_found:
        current_state = queue.popleft()
        current_id = current_state.get_state_id()
        
        # Generate all legal moves
        moves = current_state.get_legal_moves()
        
        for from_rod, to_rod, action in moves:
            new_state = current_state.apply_move(from_rod, to_rod, action)
            new_id = new_state.get_state_id()
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)
                
                # Add to search tree
                search_tree[new_id] = {
                    'state': new_state,
                    'parent': current_id,
                    'children': [],
                    'depth': new_state.depth,
                    'action': action
                }
                
                # Add as child to parent
                if current_id in search_tree:
                    search_tree[current_id]['children'].append(new_id)
                
                # Check if goal is reached
                if new_state.is_goal():
                    # Reconstruct solution path
                    node = new_id
                    while node is not None:
                        if 'action' in search_tree[node]:
                            solution_path.append(search_tree[node]['action'])
                        node = search_tree[node]['parent']
                    solution_path.reverse()
                    solution_found = True
                    break
    
    return solution_path, search_tree, visited


def print_search_tree(search_tree: Dict, max_depth: int = None):
    """Print the search tree in a readable format."""
    print("\n" + "="*60)
    print("STATE-SPACE SEARCH TREE")
    print("="*60)
    
    # Find root (state with no parent)
    root_id = None
    for state_id, info in search_tree.items():
        if info['parent'] is None:
            root_id = state_id
            break
    
    if not root_id:
        print("No root found in search tree!")
        return
    
    def print_node(state_id: str, depth: int = 0, prefix: str = ""):
        """Recursively print tree nodes."""
        if state_id not in search_tree:
            return
            
        node_info = search_tree[state_id]
        state = node_info['state']
        
        # Print current node
        indent = "  " * depth
        action = node_info.get('action', 'Initial State')
        print(f"{indent}{prefix}{action}")
        print(f"{indent}  State: {state}")
        
        # Print children
        for i, child_id in enumerate(node_info['children']):
            child_prefix = f"├─ " if i < len(node_info['children']) - 1 else "└─ "
            print_node(child_id, depth + 1, child_prefix)
    
    print_node(root_id)
    print("="*60)


def analyze_search_tree(search_tree: Dict, visited: Set):
    """Analyze and print statistics about the search."""
    print("\n" + "="*60)
    print("SEARCH ANALYSIS")
    print("="*60)
    
    total_states = len(visited)
    max_depth = 0
    branching_factors = []
    
    for state_id, info in search_tree.items():
        max_depth = max(max_depth, info['depth'])
        branching_factors.append(len(info['children']))
    
    avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
    
    print(f"Total states explored: {total_states}")
    print(f"Maximum depth: {max_depth}")
    print(f"Average branching factor: {avg_branching:.2f}")
    print(f"Minimum moves required (optimal): {2**3 - 1}")  # For 3 disks
    
    # Count states at each depth
    depth_counts = defaultdict(int)
    for info in search_tree.values():
        depth_counts[info['depth']] += 1
    
    print("\nStates at each depth:")
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]} states")

