from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Edge:
    destination: "Node"
    weight: float

    def __str__(self):
        return f" <<{self.weight}>> -> {self.destination.name}"


class Node:
    def __init__(self, name: str):
        self.name = name
        self.children: List[Edge] = []

    def add_child(self, child: "Node", weight: float) -> bool:
        if child is None:
            return False

        if self.has_child(child.name):
            return False

        self.children.append(Edge(child, weight))
        return True

    def degree(self) -> int:
        return len(self.children)

    def has_child(self, name: str) -> bool:
        return self.fetch_child(name) is not None

    def fetch_child(self, name: str) -> Optional["Node"]:
        for edge in self.children:
            if edge.destination.name == name:
                return edge.destination
        return None

    def node_edges(self) -> List[Edge]:
        return self.children

    def children_names(self) -> List[str]:
        return [edge.destination.name for edge in self.children]

    def __str__(self):
        edges = " ".join(f"|{edge}|" for edge in self.children)
        return f"{self.name} : {edges}\n"