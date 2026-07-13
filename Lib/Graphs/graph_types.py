from .node import Node , Edge
from .graph import Graph



class UndirectedGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def connect_nodes(self, parent: str, child: str, weight: float) -> bool:
        if not (self.has_node(parent) and self.has_node(child)):
            return False

        if self.fetch_node(parent).fetch_child(child):
            return False

        return (
            self.fetch_node(parent).add_child(self.fetch_node(child), weight)
            and self.fetch_node(child).add_child(self.fetch_node(parent), weight)
        )


class DirectedGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def connect_nodes(self, parent: str, child: str, weight: float) -> bool:
        if not (self.has_node(parent) and self.has_node(child)):
            return False

        if self.fetch_node(parent).fetch_child(child):
            return False

        return self.fetch_node(parent).add_child(self.fetch_node(child), weight)


class Tree(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def connect_nodes(self, parent: str, child: str, weight: float) -> bool:
        if not (self.has_node(parent) and self.has_node(child)):
            return False

        if self.fetch_node(parent).fetch_child(child):
            return False

        # Prevent cycles
        child_children = self.fetch_node(child).children_names()
        for name in child_children:
            if self.fetch_node(name).has_child(parent):
                return False

        return self.fetch_node(parent).add_child(self.fetch_node(child), weight)


class BinaryTree(Tree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def connect_nodes(self, parent: str, child: str, weight: float) -> bool:
        if not (self.has_node(parent) and self.has_node(child)):
            return False

        # Degree >= 3 because original C++ counts parent edge as well
        if self.fetch_node(parent).degree() >= 3:
            return False

        return super().connect_nodes(parent, child, weight)