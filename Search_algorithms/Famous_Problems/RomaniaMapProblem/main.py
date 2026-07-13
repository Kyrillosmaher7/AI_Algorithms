
"""
Romania map search problem demo.

Run from the project root using:

    python -m Search_Problems.Famous_Problems.RomaniaMapProblem.main
"""

from Search_algorithms.Algoritms.A_star import A_Star_Search
from Search_algorithms.Algoritms.Breadth_First_Search import Breadth_First_Search
from Search_algorithms.Algoritms.Depth_First_Search import Depth_First_Search
from Search_algorithms.Algoritms.BestFirstSearch import Best_First_Search
from Search_algorithms.Algoritms.Dijikstra import Dijkstra
from Lib.Graphs.graph_types import UndirectedGraph 

Map = UndirectedGraph() 
Map.add_node("Arad") 
Map.add_node("Zerind") 
Map.add_node("Oradea") 
Map.add_node("Sibiu") 
Map.add_node("Timisoara") 
Map.add_node("Lugoj") 
Map.add_node("Mehadia") 
Map.add_node("Drobeta") 
Map.add_node("Craiova") 
Map.add_node("Rimnicu Vilcea") 
Map.add_node("Fagaras") 
Map.add_node("Pitesti") 
Map.add_node("Bucharest") 
Map.add_node("Giurgiu") 
Map.add_node("Urziceni") 
Map.add_node("Hirsova") 
Map.add_node("Eforie") 
Map.add_node("Vaslui") 
Map.add_node("Iasi") 
Map.add_node("Neamt") 
Map.connect_nodes("Arad", "Zerind", 75) 
Map.connect_nodes("Arad", "Sibiu", 140) 
Map.connect_nodes("Arad", "Timisoara", 118) 
Map.connect_nodes("Zerind", "Oradea", 71) 
Map.connect_nodes("Oradea", "Sibiu", 151) 
Map.connect_nodes("Sibiu", "Fagaras", 99)   
Map.connect_nodes("Sibiu", "Rimnicu Vilcea", 80) 
Map.connect_nodes("Timisoara", "Lugoj", 111) 
Map.connect_nodes("Lugoj", "Mehadia", 70) 
Map.connect_nodes("Mehadia", "Drobeta", 75) 
Map.connect_nodes("Drobeta", "Craiova", 120) 
Map.connect_nodes("Craiova", "Rimnicu Vilcea", 146) 
Map.connect_nodes("Rimnicu Vilcea", "Pitesti", 97) 
Map.connect_nodes("Craiova", "Pitesti", 138) 
Map.connect_nodes("Fagaras", "Bucharest", 211) 
Map.connect_nodes("Pitesti", "Bucharest", 101) 
Map.connect_nodes("Bucharest", "Giurgiu", 90) 
Map.connect_nodes("Bucharest", "Urziceni", 85) 
Map.connect_nodes("Urziceni", "Hirsova", 98) 
Map.connect_nodes("Hirsova", "Eforie", 86) 
Map.connect_nodes("Urziceni", "Vaslui", 142) 
Map.connect_nodes("Vaslui", "Iasi", 92) 
Map.connect_nodes("Iasi", "Neamt", 87) 
def heuristic(node_name: str) -> float:
    heuristic_values = {
        "Arad": 366,
        "Bucharest": 0,
        "Craiova": 160,
        "Drobeta": 242,
        "Eforie": 161,
        "Fagaras": 176,
        "Giurgiu": 77,
        "Hirsova": 151,
        "Iasi": 226,
        "Lugoj": 244,
        "Mehadia": 241,
        "Neamt": 234,
        "Oradea": 380,
        "Pitesti": 100,
        "Rimnicu Vilcea": 193,
        "Sibiu": 253,
        "Timisoara": 329,
        "Urziceni": 80,
        "Vaslui": 199,
        "Zerind": 374
    }
    return heuristic_values.get(node_name, float('inf'))

# Solvimg the problem with different search algorithms
start_node = "Arad"
goal_node = "Bucharest"

#------------------------------------------ Using Breadth First Search Algorithm ------------------------------------------
Path , Goal, Cost = Breadth_First_Search(Map, start_node, goal_node)
print("BFS Solution:", Path)
print("BFS Total Cost:", Cost)
print("\n************************************************\n")
#------------------------------------------ Using Depth First Search Algorithm ------------------------------------------
Path , Goal, Cost = Depth_First_Search(Map, start_node, goal_node)
print("DFS Solution:", Path)
print("DFS Total Cost:", Cost)
print("\n************************************************\n")
#------------------------------------------ Using Dijkstra's Algorithm ------------------------------------------
Path , Cost  = Dijkstra(Map, start_node, goal_node)
print("Dijkstra's Solution:", Path)
print("Dijkstra's Total Cost:", Cost)
print("\n************************************************\n")
#------------------------------------------ Using Best First Search Algorithm ------------------------------------------
Path , Cost = Best_First_Search(Map, start_node, goal_node, heuristic)
print("Best First Search Solution:", Path)
print("Best First Search Total Cost:", Cost)
print("\n************************************************\n")
#------------------------------------------ Using A* Search Algorithm ------------------------------------------
Path , Cost = A_Star_Search(Map, start_node, goal_node, heuristic)
print("A* Search Solution:", Path)
print("A* Search Total Cost:", Cost)
print("\n ************************************************------- \n")