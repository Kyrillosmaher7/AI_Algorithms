from collections import deque
from Lib.Graphs.graph import Graph
from typing import List, Tuple, Optional

def Breadth_First_Search(graph: Graph, start_name: str, goal: str) -> Tuple[List[str], Optional[str], float]:
    """
    Performs Breadth-First Search on a weighted graph and returns the path cost.
    
    Args:
        graph: The weighted graph to search
        start_name: Starting node name
        goal: Goal node name
        
    Returns:
        Tuple containing (path from start to goal, goal_node_name, total_cost)
        Returns ([], None, 0.0) if path not found
    """
    start = graph.fetch_node(start_name)
    if not start:
        return [], None, 0.0
    
    # If start is already the goal
    if start_name == goal:
        return [start_name], goal, 0.0
    
    # BFS initialization
    visited = set([start_name])
    queue = deque([start])
    
    # For path reconstruction and cost tracking
    parent = {start_name: None}
    # Track cumulative cost to reach each node
    cost_to_reach = {start_name: 0.0}
    
    while queue:
        current = queue.popleft()
        current_cost = cost_to_reach[current.name]
        
        # Explore neighbors through edges 
        for edge in current.children:  # Directly access the edges list
            neighbor_node = edge.destination
            neighbor_name = neighbor_node.name
            
            # Get weight from the edge
            weight = edge.weight
            new_cost = current_cost + weight
            
            if neighbor_name not in visited:
                visited.add(neighbor_name)
                parent[neighbor_name] = current.name
                cost_to_reach[neighbor_name] = new_cost
                queue.append(neighbor_node)
                
                # Check if we found the goal
                if neighbor_name == goal:
                    # Reconstruct path
                    path = []
                    node = neighbor_name
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path, goal, new_cost
    
    # Goal not found
    return [], None, 0.0