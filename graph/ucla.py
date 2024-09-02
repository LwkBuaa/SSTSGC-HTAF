import sys
sys.path.extend(['../'])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)]

parallelward_3 = [[4, 5, 2, 9],
                   [4, 6, 1, 10],
                   [4, 7, 13, 17, 11],
                   [4, 8, 14, 18, 12],
                   [4, 8, 15, 19, 12],
                   [4, 8, 16, 20, 12]]

parallelward_2 = [[4, 5, 13, 17, 9],
                  [4, 6, 14, 18, 10],
                  [4, 7, 15, 19, 11],
                  [4, 8, 16, 20, 12]]

parallelward_1 = [[2, 13, 17],
                  [3, 14, 18],
                  [4, 5, 15, 19, 9],
                  [4, 6, 16, 20, 10],
                  [4, 7, 16, 20, 11],
                  [4, 8, 16, 20, 12]]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, com=3):
        self.num_node = num_node
        self.self_link = self_link
        self.parallelward_3 = parallelward_3
        self.parallelward_2 = parallelward_2
        self.parallelward_1 = parallelward_1
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(com)

    def get_adjacency_matrix(self, com=3):
        if com == 3:
            parallelward = self.parallelward_3
        elif com == 2:
            parallelward = self.parallelward_2
        elif com == 1:
            parallelward = self.parallelward_1
        else:
            raise ValueError()
        A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward, parallelward)

        return A
