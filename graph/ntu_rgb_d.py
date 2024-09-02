import sys
sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
                    ]
parallelward_21 = [[2, 9, 3, 5],
                   [1, 10, 4, 6],
                   [17, 11, 4, 7, 13],
                   [18, 12, 4, 8, 14],
                   [19, 25, 4, 23, 15],
                   [20, 24, 4, 22, 16]]

parallelward_2 = [[3, 9, 17, 13, 5],
                  [4, 10, 18, 14, 6],
                  [4, 11, 19, 15, 7],
                  [4, 12, 20, 16, 8],
                  [4, 25, 20, 16, 23],
                  [4, 24, 20, 16, 22]]

parallelward_1 = [[2, 17, 13],
                  [21, 18, 14],
                  [3, 9, 19, 15, 5],
                  [4, 10, 20, 16, 6],
                  [4, 11, 20, 16, 7],
                  [4, 12, 20, 16, 8],
                  [4, 25, 20, 16, 23],
                  [4, 24, 20, 16, 22]]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, com='21'):
        self.num_node = num_node
        self.self_link = self_link
        self.parallelward_21 = parallelward_21
        self.parallelward_2 = parallelward_2
        self.parallelward_1 = parallelward_1
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(com)

    def get_adjacency_matrix(self, com=21):
        if com == 21:
            parallelward = self.parallelward_21
        elif com == 2:
            parallelward = self.parallelward_2
        elif com == 1:
            parallelward = self.parallelward_1
        else:
            raise ValueError()
        A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward, parallelward)

        return A
