import numpy as np


def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An


def normalize_digraph(A):
    Dl = np.sum(A, 1)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(Dn, A)
    return AD


def web2digraph(parallelward, num_node):
    A = np.zeros((num_node, num_node))
    for j in range(len(parallelward)):
        for index_i, i in enumerate(parallelward[j]):
            if index_i == 0:
                A[i - 1, parallelward[j][-1] - 1] = 1
                A[i - 1, parallelward[j][index_i + 1] - 1] = 1
            elif index_i == len(parallelward[j]) - 1:
                A[i - 1, parallelward[j][0] - 1] = 1
                A[i - 1, parallelward[j][index_i - 1] - 1] = 1
            else:
                A[i - 1, parallelward[j][index_i + 1] - 1] = 1
                A[i - 1, parallelward[j][index_i - 1] - 1] = 1
    return A


def get_spatial_graph(num_node, self_link, inward, outward, parallelward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    Parallel = normalize_digraph(web2digraph(parallelward, num_node))
    A = np.stack((I, In, Out, Parallel))
    return A


def get_ins_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(np.ones_like(I) - I)
    Out = In.T
    A = np.stack((I, In, Out))
    return A


def get_spatial_graphnextv2(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    # In = normalize_digraph(edge2mat(inward, num_node))
    # Out = normalize_digraph(edge2mat(outward, num_node))
    # SELF = np.eye(num_node)
    A = np.stack((I, I, I, I))
    return A


def get_spatial_graphnext(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    SELF = np.eye(num_node)
    A = np.stack((I, In, Out, SELF))
    return A


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
         - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A
