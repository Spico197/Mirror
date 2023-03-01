from collections import defaultdict

import torch
from rex.utils.iteration import windowed_queue_iter


def find_paths_from_adj_mat(adj_mat: torch.Tensor) -> list[tuple[int]]:
    assert adj_mat.shape[0] == adj_mat.shape[1] and len(adj_mat.shape) == 2

    paths = []
    self_loops = set()
    adj_map = defaultdict(set)
    rev_adj_map = defaultdict(set)
    # current -> next
    for c, n in adj_mat.detach().nonzero().tolist():
        # self-loop
        if c == n:
            self_loops.add(c)
        else:
            adj_map[c].add(n)
            # reversed map
            rev_adj_map[n].add(c)
    for self_loop_node in self_loops:
        paths.append((self_loop_node,))

    def track(path: tuple[int], c: int, visited: set[tuple[int]]):
        if c in adj_map:
            for n in adj_map[c]:
                if (c, n) in visited:
                    continue
                visited.add((c, n))
                track(path + (c,), n, visited)
        else:
            if path:
                paths.append(path + (c,))

        # # include loops
        # if path not in paths and all(not set(path).issubset(p) for p in paths):
        #     paths.append(path)

    start_nodes = set(adj_map.keys()) - set(rev_adj_map.keys())
    for c in start_nodes:
        ns = adj_map[c]
        for n in ns:
            track((c,), n, set())

    return paths


def encode_nnw_thw_matrix(
    spans: list[tuple[int]], seq_len: int, nnw_id: int = 0, thw_id: int = 1
) -> torch.Tensor:
    mat = torch.zeros(seq_len, seq_len, 2)
    for span in spans:
        for s, e in windowed_queue_iter(span, 2, 1, drop_last=True):
            mat[s, e, nnw_id] = 1
        mat[span[-1], span[0], thw_id] = 1
    return mat


def decode_nnw_thw_mat(
    batch_mat: torch.LongTensor, nnw_id: int = 0, thw_id: int = 1
) -> list[list[tuple[int]]]:
    """Decode NNW THW matrix into a list of spans

    Args:
        matrix: (batch_size, seq_len, seq_len, 2)
    """
    ins_num, seq_len1, seq_len2, cls_num = batch_mat.shape
    assert seq_len1 == seq_len2
    assert cls_num == 2

    result_batch = []
    for ins_id in range(ins_num):
        ins_spans = []
        # ins_mat: (seq_len, seq_len, 2)
        ins_mat = batch_mat[ins_id]
        nnw_paths = find_paths_from_adj_mat(ins_mat[..., nnw_id])
        thw_pairs = ins_mat[..., thw_id].detach().nonzero().tolist()
        # reversed match, end -> start
        for e, s in thw_pairs:
            for path in nnw_paths:
                if path[-1] == e and path[0] == s:
                    ins_spans.append(path)
        result_batch.append(ins_spans)

    return result_batch
