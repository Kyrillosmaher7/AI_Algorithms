from typing import Dict, List, Optional
from .node import Node


class Graph:
    def __init__(self): 
        self.nodes: Dict[str, Node] = {}

    def add_node(self, name: str) -> bool:
        if self.has_node(name):
            return False

        self.nodes[name] = Node(name)
        return True

    def has_node(self, name: str) -> bool:
        return name in self.nodes

    def fetch_node(self, name: str) -> Optional[Node]:
        return self.nodes.get(name)

    def vertices(self) -> List[str]:
        return list(self.nodes.keys())

    def __str__(self):
        result = []
        for name in self.vertices():
            result.append(str(self.fetch_node(name)))
        return "".join(result)