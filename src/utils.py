from collections import defaultdict

import torch
from rex.utils.iteration import windowed_queue_iter
from rex.utils.position import find_all_positions


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

    def track(path: tuple[int], c: int):
        visited: set[tuple[int]] = set()
        stack = [(path, c)]
        while stack:
            path, c = stack.pop()
            if c in adj_map:
                for n in adj_map[c]:
                    if (c, n) in visited:
                        continue
                    visited.add((c, n))
                    stack.append((path + (c,), n))
            # else:
            if path:
                paths.append(path + (c,))

    # def track(path: tuple[int], c: int, visited: set[tuple[int]]):
    #     if c in adj_map:
    #         for n in adj_map[c]:
    #             if (c, n) in visited:
    #                 continue
    #             visited.add((c, n))
    #             track(path + (c,), n, visited)
    #     else:
    #         if path:
    #             paths.append(path + (c,))

    #     # # include loops
    #     # if path not in paths and all(not set(path).issubset(p) for p in paths):
    #     #     paths.append(path)

    start_nodes = set(adj_map.keys()) - set(rev_adj_map.keys())
    for c in start_nodes:
        ns = adj_map[c]
        for n in ns:
            track((c,), n)

    return paths


def encode_nnw_thw_mat(
    spans: list[tuple[int]], seq_len: int, nnw_id: int = 0, thw_id: int = 1
) -> torch.Tensor:
    mat = torch.zeros(2, seq_len, seq_len)
    for span in spans:
        if len(span) == 1:
            mat[:, span[0], span[0]] = 1
        else:
            for s, e in windowed_queue_iter(span, 2, 1, drop_last=True):
                mat[nnw_id, s, e] = 1
        mat[thw_id, span[-1], span[0]] = 1
    return mat


def decode_nnw_thw_mat(
    batch_mat: torch.LongTensor,
    nnw_id: int = 0,
    thw_id: int = 1,
    offsets: list[int] = None,
) -> list[list[tuple[int]]]:
    """Decode NNW THW matrix into a list of spans

    Args:
        matrix: (batch_size, 2, seq_len, seq_len)
    """
    ins_num, cls_num, seq_len1, seq_len2 = batch_mat.shape
    assert seq_len1 == seq_len2
    assert cls_num == 2

    result_batch = []
    for ins_id in range(ins_num):
        offset = offsets[ins_id] if offsets else 0
        ins_span_paths = []
        # ins_mat: (2, seq_len, seq_len)
        ins_mat = batch_mat[ins_id]
        nnw_paths = find_paths_from_adj_mat(ins_mat[nnw_id, ...])
        end_start_to_paths = defaultdict(set)
        for path in nnw_paths:
            end_start_to_paths[(path[-1], path[0])].add(path)
        thw_pairs = ins_mat[thw_id, ...].detach().nonzero().tolist()
        # reversed match, end -> start
        for e, s in thw_pairs:
            for path in end_start_to_paths[(e, s)]:
                ins_span_paths.append(tuple(i - offset for i in path))
        result_batch.append(ins_span_paths)

    return result_batch


def decode_pointer_mat(
    batch_mat: torch.LongTensor, offsets: list[int] = None
) -> list[list[tuple[int]]]:
    batch_paths = []
    for i in range(len(batch_mat)):
        offset = offsets[i] if offsets else 0
        coordinates = (batch_mat[i, 0] == 1).nonzero().tolist()
        paths = []
        for s, e in coordinates:
            path = tuple(range(s - offset, e + 1 - offset))
            paths.append(path)
        batch_paths.append(paths)
    return batch_paths


def encode_nnw_nsw_thw_mat(
    spans: list[list[tuple[int]]],
    seq_len: int,
    nnw_id: int = 0,
    nsw_id: int = 1,
    thw_id: int = 2,
) -> torch.Tensor:
    mat = torch.zeros(3, seq_len, seq_len)
    for parts in spans:
        span = ()
        for p_i, part in enumerate(parts):
            if not all(0 <= el <= seq_len - 1 for el in part):
                continue
            span += part
            if p_i < len(parts) - 1 and 0 <= parts[p_i + 1][0] <= seq_len - 1:
                # current part to next part
                mat[nsw_id, parts[p_i][-1], parts[p_i + 1][0]] = 1
        if len(span) == 1:
            mat[:, span[0], span[0]] = 1
        elif len(span) > 1:
            for s, e in windowed_queue_iter(span, 2, 1, drop_last=True):
                mat[nnw_id, s, e] = 1
        if span:
            mat[thw_id, span[-1], span[0]] = 1
    return mat


def split_tuple_by_positions(nums, positions) -> list:
    """
    Examples:
        >>> nums = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        >>> positions = [2, 5, 7]
        >>> split_tuple_by_positions(nums, positions)
        ((1, 2), (3, 4, 5), (6, 7), (8, 9, 10))
    """
    # Check if the given positions are valid
    if not all(p < len(nums) for p in positions):
        raise ValueError("Invalid positions")

    # Add 0 and len(nums) to the list of positions
    positions = [0] + sorted(positions) + [len(nums)]

    # Split the tuple into multiple tuples based on the positions
    result = []
    for i in range(1, len(positions)):
        start = positions[i - 1]
        end = positions[i]
        result.append(nums[start:end])

    return result


def decode_nnw_nsw_thw_mat(
    batch_mat: torch.LongTensor,
    nnw_id: int = 0,
    nsw_id: int = 1,
    thw_id: int = 2,
    offsets: list[int] = None,
) -> list[list[tuple[int]]]:
    """Decode NNW NSW THW matrix into a list of spans
    One span has multiple parts

    Args:
        batch_mat: (batch_size, 3, seq_len, seq_len)
    """
    ins_num, cls_num, seq_len1, seq_len2 = batch_mat.shape
    assert seq_len1 == seq_len2
    assert cls_num == 3

    result_batch = []
    for ins_id in range(ins_num):
        offset = offsets[ins_id] if offsets else 0
        ins_span_paths = set()
        # ins_mat: (2, seq_len, seq_len)
        ins_mat = batch_mat[ins_id]
        nsw_connections = {
            (part1e, part2s)
            for part1e, part2s in ins_mat[nsw_id, ...].detach().nonzero().tolist()
        }
        nnw_paths = find_paths_from_adj_mat(ins_mat[nnw_id, ...])
        end_start_to_paths = defaultdict(set)
        for path in nnw_paths:
            end_start_to_paths[(path[-1], path[0])].add(path)
        thw_pairs = ins_mat[thw_id, ...].detach().nonzero().tolist()
        # reversed match, end -> start
        for e, s in thw_pairs:
            for path in nnw_paths:
                if s in path:
                    sub_path = path[path.index(s) :]
                    if e in sub_path:
                        sub_path = sub_path[: sub_path.index(e) + 1]
                        chain = tuple(i - offset for i in sub_path)
                        parts = []
                        all_sep_positions = set()
                        # cut path into multiple spans if there are skip links
                        if len(chain) > 1:
                            for sep in nsw_connections:
                                sep = tuple(i - offset for i in sep)
                                positions = find_all_positions(list(chain), list(sep))
                                if positions:
                                    # +1: (5, 6, 269) with (6, 269) as sep, found position is 1,
                                    # while we want to split after 6, which needs +1
                                    positions = {p[0] + 1 for p in positions}
                                    all_sep_positions.update(positions)
                            parts = split_tuple_by_positions(chain, all_sep_positions)
                        if not parts:
                            parts = [chain]
                        ins_span_paths.add(tuple(parts))
        result_batch.append(list(ins_span_paths))

    return result_batch


# def encode_nnw_nsw_thw_mat(
#     spans: list[list[tuple[int]]],
#     seq_len: int,
#     nnw_id: int = 0,
#     nsw_id: int = 1,
#     thw_id: int = 2,
# ) -> torch.Tensor:
#     mat = torch.zeros(3, seq_len, seq_len)
#     for span in spans:
#         for p_i, part in enumerate(span):
#             if len(part) == 1:
#                 mat[:, part[0], part[0]] = 1
#             else:
#                 for s, e in windowed_queue_iter(part, 2, 1, drop_last=True):
#                     mat[nnw_id, s, e] = 1
#             if p_i < len(span) - 1:
#                 # current part to next part
#                 mat[nsw_id, span[p_i][-1], span[p_i + 1][0]] = 1
#         mat[thw_id, span[-1][-1], span[0][0]] = 1
#     return mat


# def decode_nnw_nsw_thw_mat(
#     batch_mat: torch.LongTensor,
#     nnw_id: int = 0,
#     nsw_id: int = 1,
#     thw_id: int = 2,
#     offsets: list[int] = None,
# ) -> list[list[tuple[int]]]:
#     """Decode NNW NSW THW matrix into a list of spans
#     One span has multiple parts

#     Args:
#         batch_mat: (batch_size, 3, seq_len, seq_len)
#     """

#     ins_num, cls_num, seq_len1, seq_len2 = batch_mat.shape
#     assert seq_len1 == seq_len2
#     assert cls_num == 2

#     result_batch = []
#     for ins_id in range(ins_num):
#         offset = offsets[ins_id] if offsets else 0
#         ins_span_paths = []
#         # ins_mat: (3, seq_len, seq_len)
#         ins_mat = batch_mat[ins_id]
#         nnw_paths = find_paths_from_adj_mat(ins_mat[nnw_id, ...])

#         path_index = {"s": defaultdict(set), "e": defaultdict(set)}
#         for path in nnw_paths:
#             s = path[0]
#             e = path[-1]
#             path_index["s"][s].add(path)
#             path_index["e"][e].add(path)

#         nsw_connections = {(part1e, part2s) for part1e, part2s in ins_mat[nsw_id, ...].detach().nonzero().tolist()}
#         thw_connections = {(span_e, span_s) for span_e, span_s in ins_mat[thw_id, ...].detach().nonzero().tolist()}
#         for e, s in thw_connections:


#         path_span_combinations = []
#         for part1_e, part2_s in nsw_connections:
#             part1s = path_index["e"][part1_e]
#             part2s = path_index["s"][part2_s]
#             # for part1 in part1s:
#             #     for part2 in part2s:
#             #         if ()

#         end_start_to_paths = defaultdict(set)
#         for path in nnw_paths:
#             end_start_to_paths[(path[-1], path[0])].add(path)
#         # reversed match, end -> start
#         for e, s in thw_pairs:
#             for path in end_start_to_paths[(e, s)]:
#                 ins_span_paths.append(tuple(i - offset for i in path))
#         result_batch.append(ins_span_paths)

#     return result_batch
