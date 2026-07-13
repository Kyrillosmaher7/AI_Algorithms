import heapq
from Lib.Graphs.graph import Graph
from typing import List, Tuple, Optional, Callable

def Best_First_Search(graph: Graph, start: str, goal: str, heuristic: Callable[[str], float]) -> Tuple[List[str], float]:
    """
    Performs Best-First Search (Greedy Best-First Search) on a graph.
    
    Args:
        graph: The graph to search
        start: The starting node name
        goal: The goal node name
        heuristic: A function that takes a node name and returns its heuristic value
    
    Returns:
        Tuple containing (path from start to goal, total cumulative cost)
        Returns ([], 0.0) if path not found
    """
    # Check if start and goal exist
    if not graph.has_node(start) or not graph.has_node(goal):
        return [], 0.0
    
    # If start is already the goal
    if start == goal:
        return [start], 0.0
    
    # Priority queue for nodes to explore (min-heap based on heuristic)
    frontier = []
    heapq.heappush(frontier, (heuristic(start), start))
    
    # Track visited nodes, their parents, and cumulative cost
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    visited = set([start])
    
    while frontier:
        # Get the node with the lowest heuristic value
        _, current = heapq.heappop(frontier)
        
        # Check if we found the goal
        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, cost_so_far[goal]
        
        # Get current node object
        current_node = graph.fetch_node(current)
        if not current_node:
            continue
            
        # Explore neighbors
        for edge in current_node.children:
            neighbor = edge.destination.name
            # edge.weight is used to calculate the real cost
            weight = edge.weight 
            
            # Skip if already visited
            if neighbor in visited:
                continue
                
            visited.add(neighbor)
            came_from[neighbor] = current
            # Update the real cumulative cost to reach this neighbor
            cost_so_far[neighbor] = cost_so_far[current] + weight
            
            # Add to frontier with heuristic value for expansion priority
            heapq.heappush(frontier, (heuristic(neighbor), neighbor))
    
    # Goal not found
    return [], 0.0