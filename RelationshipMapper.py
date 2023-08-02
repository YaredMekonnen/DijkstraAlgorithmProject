import heapq
import math
import plotly.graph_objects as go


class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def add_person(self, name):
        if name not in self.vertices:
            self.vertices[name] = []

    def add_relationship(self, person1, relationship, person2, weight):
        self.add_person(person1)
        self.add_person(person2)
        self.vertices[person1].append((person2, weight))
        self.vertices[person2].append((person1, weight))
        self.edges[(person1, person2)] = weight
        self.edges[(person2, person1)] = weight

    def display_tree(self):
        fig = go.Figure()
        pos = self.get_node_positions()
        for edge in self.edges:
            x = [pos[edge[0]][0], pos[edge[1]][0], None]
            y = [pos[edge[0]][1], pos[edge[1]][1], None]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='rgb(210,210,210)', width=1)))
        for name in pos:
            x_pos, y_pos, edges = pos[name]
            fig.add_trace(go.Scatter(x=[x_pos], y=[y_pos], mode='markers+text', marker=dict(size=20), text=name, textposition='top center'))
        fig.update_layout(showlegend=False)
        fig.show()

    def get_node_positions(self):
        if not self.vertices:
            return {}

        root = next(iter(self.vertices.keys()))

        visited = set()
        node_positions = {}

        def dfs(node, level, x_pos):
            if node in visited:
                return
            visited.add(node)

            # Calculate y position based on level
            y_pos = level * -1

            # Store node position
            node_positions[node] = (x_pos, y_pos, [edge[1] for edge in self.vertices[node]])

            # Recursively visit children
            children = self.vertices[node]
            num_children = len(children)
            if num_children > 1:
                space = 2 / (num_children - 1)
                x_offset = -1
                for child, weight in children:
                    dfs(child, level - 1, x_pos + x_offset)
                    x_offset += space
            elif num_children == 1:
                dfs(children[0][0], level - 1, x_pos)

        dfs(root, 0, 0)
        return node_positions

    # def dijkstra(self, start):
    #     distances = {node: math.inf for node in self.vertices}
    #     distances[start] = 0
    #     heap = [(0, start)]
    #     visited = set()

    #     while heap:
    #         (distance, current_node) = heapq.heappop(heap)
    #         if current_node in visited:
    #             continue
    #         visited.add(current_node)

    #         for neighbor, weight in self.vertices[current_node]:
    #             path_cost = distances[current_node] + weight
    #             if path_cost < distances[neighbor]:
    #                 distances[neighbor] = path_cost
    #                 heapq.heappush(heap, (path_cost, neighbor))

    #     return distances
    def dijkstra(self, start):
        distances = {node: math.inf for node in self.vertices}
        distances[start] = 0
        heap = [(0, start)]
        visited = set()
        paths = {start: [start]}

        while heap:
            (distance, current_node) = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)

            for neighbor, weight in self.vertices[current_node]:
                path_cost = distances[current_node] + weight
                if path_cost < distances[neighbor]:
                    distances[neighbor] = path_cost
                    heapq.heappush(heap, (path_cost, neighbor))
                    paths[neighbor] = paths[current_node] + [neighbor]

        return distances, paths





# Add people
g = Graph()

# Add users
g.add_person("Alice")
g.add_person("Bob")
g.add_person("Charlie")
g.add_person("David")
g.add_person("Eve")
g.add_person("Frank")
g.add_person("Hannah")
g.add_person("Isaac")
g.add_person("Jacob")
g.add_person("Kate")
g.add_person("Laura")
g.add_person("Maggie")
g.add_person("Nate")
g.add_person("Olivia")
g.add_person("Peter")
g.add_person("Quinn")
g.add_person("Rachel")
g.add_person("Sam")

# Add relationships
g.add_relationship("Alice", "married to", "Bob",5)
g.add_relationship("Bob", "married to", "Eve",4)
g.add_relationship("Alice", "parent of", "Charlie",6)
g.add_relationship("Eve", "parent of", "Charlie",7)
g.add_relationship("Alice", "parent of", "David",8)
g.add_relationship("Bob", "parent of", "David",9)
g.add_relationship("Eve", "parent of", "Frank",5)
g.add_relationship("Frank", "married to", "Hannah",3)
g.add_relationship("Frank", "parent of", "Isaac",5)
g.add_relationship("Hannah", "parent of", "Isaac",2)
g.add_relationship("Isaac", "married to", "Maggie",7)
g.add_relationship("Isaac", "parent of", "Jacob",8)
g.add_relationship("Maggie", "parent of", "Jacob",9)
g.add_relationship("Jacob", "married to", "Kate",8)
g.add_relationship("Jacob", "parent of", "Laura",9)
g.add_relationship("Kate", "parent of", "Laura",10)
g.add_relationship("Laura", "married to", "Nate",6)
g.add_relationship("Laura", "parent of", "Olivia",80)
g.add_relationship("Nate", "parent of", "Olivia",12)
g.add_relationship("Olivia", "married to", "Peter",11)
g.add_relationship("Olivia", "parent of", "Quinn",9)
g.add_relationship("Peter", "parent of", "Quinn",90)
g.add_relationship("Quinn", "married to", "Rachel",78)
g.add_relationship("Quinn", "parent of", "Sam",90)
g.add_relationship("Rachel", "parent of", "Sam",7)



# Find the shortest path between Alice and David


# Display family tree
g.display_tree()


# Define a graph
# g = Graph()

# # Add nodes
# g.add_person('A')
# g.add_person('B')
# g.add_person('C')
# g.add_person('D')
# g.add_person('E')

# # Add edges with weights
# g.add_relationship('A', 'connects_to', 'B', 3)
# g.add_relationship('A', 'connects_to', 'C', 4)
# g.add_relationship('B', 'connects_to', 'C', 2)
# g.add_relationship('B', 'connects_to', 'D', 5)
# g.add_relationship('C', 'connects_to', 'D', 1)
# g.add_relationship('C', 'connects_to', 'E', 6)
# g.add_relationship('D', 'connects_to', 'E', 7)
# g.display_tree()
# Find shortest path from A to E
# shortest_distances = g.dijkstra('A')
# print(shortest_distances['E'])  # Output: 10

distances, paths = g.dijkstra('Sam')
for node in distances:
    path = paths[node]
    print(f'Shortest path from Alice to {node}: {" -> ".join(path)}, distance: {distances[node]}')
