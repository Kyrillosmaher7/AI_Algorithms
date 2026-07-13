from Lib.Graphs.graph  import Graph
import heapq
from typing import List, Tuple, Optional

def Dijkstra(graph: Graph, start_name: str, goal: str) -> Tuple[List[str], Optional[float]]:
    """
    Dijkstra's algorithm for finding the shortest path in a graph with non-negative weights.
    
    Args:
        graph: The graph to search
        start_name: Starting node name
        goal: Goal node name
        
    Returns:
        Tuple containing (path from start to goal, total cost)
        Returns ([], None) if path not found
    """
    # Check if start and goal exist
    if not graph.has_node(start_name) or not graph.has_node(goal):
        return [], None
    
    # Initialize distances and previous nodes
    distances = {node: float('inf') for node in graph.vertices()}
    previous = {node: None for node in graph.vertices()}
    distances[start_name] = 0
    
    # Priority queue for exploring nodes (min-heap)
    priority_queue = [(0, start_name)]  # (distance, node_name)
    
    # Track visited nodes to avoid reprocessing
    visited = set()

    while priority_queue:
        current_distance, current_node_name = heapq.heappop(priority_queue)

        # Skip if we already found a better path to this node
        if current_distance > distances[current_node_name]:
            continue
            
        # If we reached the goal, reconstruct the path
        if current_node_name == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = previous[node]
            path.reverse()
            return path, distances[goal]

        # Mark as visited 
        visited.add(current_node_name)

        current_node = graph.fetch_node(current_node_name)
        if not current_node:
            continue

        # Explore neighbors through edges
        for edge in current_node.children:  # Directly access the edges list
            neighbor_node = edge.destination
            neighbor_name = neighbor_node.name
            
            # Skip if already visited
            if neighbor_name in visited:
                continue
                
            # Get weight from the edge
            weight = edge.weight
            distance = current_distance + weight

            # Only consider this new path if it's better
            if distance < distances[neighbor_name]:
                distances[neighbor_name] = distance
                previous[neighbor_name] = current_node_name
                heapq.heappush(priority_queue, (distance, neighbor_name))
                
    return [], None  # Goal not found