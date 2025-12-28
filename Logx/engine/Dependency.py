from collections import defaultdict

class DependencyGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_dependency(self, file: str, depends_on: str):
        if depends_on:
            self.graph[file].append(depends_on)

    def get_execution_order(self):
        visited = set()
        order = []
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for dep in self.graph.get(node, []):
                dfs(dep)
            order.append(node)
        for node in self.graph:
            dfs(node)
        return order[::-1]
