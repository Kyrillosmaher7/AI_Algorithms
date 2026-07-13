import heapq
from Lib.Graphs.graph import Graph
from typing import List, Tuple, Optional, Callable

def A_Star_Search(graph: Graph, start: str, goal: str, heuristic: Callable[[str], float]) -> Tuple[List[str], float]:
    """
    Performs A* Search on a graph.
    Args:
        graph: The graph to search
        start: The starting node name
        goal: The goal node name
        heuristic: A function that takes a node name and returns its heuristic value
    
    Returns:
        Tuple containing (path from start to goal, total cumulative cost)
    """
    if not graph.has_node(start) or not graph.has_node(goal):
        return [], 0.0

    # frontier stores (f_score, current_node_name)
    frontier = []
    heapq.heappush(frontier, (heuristic(start), start))
    
    came_from = {start: None}
    # g_score: the cost of the cheapest path from start to n currently known
    g_score = {start: 0.0}
    
    while frontier:
        # Get node with the lowest f_score (g + h)
        _, current = heapq.heappop(frontier)
        
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, g_score[goal]
        
        current_node = graph.fetch_node(current)
        if not current_node:
            continue
            
        for edge in current_node.children:
            neighbor = edge.destination.name
            # Tentative g_score is the cost from start to neighbor through current
            tentative_g_score = g_score[current] + edge.weight
            
            # If this path to neighbor is better than any previous one, record it
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor)
                heapq.heappush(frontier, (f_score, neighbor))
    
    return [], 0.0