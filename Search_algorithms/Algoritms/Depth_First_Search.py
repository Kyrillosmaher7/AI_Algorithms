from collections import deque
from Lib.Graphs.graph import Graph
from typing import List, Tuple, Optional

def Depth_First_Search(graph: Graph, start_name: str, goal: str) -> Tuple[List[str], Optional[str], float]:
    """
    Performs Depth-First Search on a weighted graph and returns the path cost.
    
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
    
    # DFS initialization - stack stores (node, path_to_node, cumulative_cost)
    visited = set()
    stack = [(start, [start_name], 0.0)]  # (node, path_to_node, current_cost)
    
    while stack:
        current, path, current_cost = stack.pop()
        
        if current.name == goal:
            return path, goal, current_cost
        
        if current.name not in visited:
            visited.add(current.name)
            
            # Explore neighbors through edges 
            for edge in current.children:  # Access edges list directly
                neighbor_node = edge.destination
                neighbor_name = neighbor_node.name
                
                if neighbor_name not in visited:
                    # Get weight from the edge
                    weight = edge.weight
                    new_cost = current_cost + weight
                    stack.append((neighbor_node, path + [neighbor_name], new_cost))
    
    # Goal not found
    return [], None, 0.0