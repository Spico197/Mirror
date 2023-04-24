import torch

from src.utils import decode_nnw_thw_mat, encode_nnw_thw_mat, find_paths_from_adj_mat


def test_find_path_from_adj_mat():
    adj_mat = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ]
    adj_mat = torch.tensor(adj_mat).byte()
    paths = find_paths_from_adj_mat(adj_mat)
    assert set(paths) == {(0, 1, 2, 5, 4, 3)}

    adj_mat = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    adj_mat = torch.tensor(adj_mat).byte()
    paths = find_paths_from_adj_mat(adj_mat)
    assert set(paths) == {(3,)}


def test_encode_decode_nnw_thw_mat():
    spans = [(0, 1, 2, 5, 4, 3), (0,)]
    adj_mat = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.float,
    )
    mat = encode_nnw_thw_mat(spans, max(spans[0]) + 1)
    assert torch.equal(mat[0, ...], adj_mat)
    thw_mat = torch.zeros_like(adj_mat)
    thw_mat[3, 0] = 1
    thw_mat[0, 0] = 1
    assert torch.equal(mat[1, ...], thw_mat)
    decoded = decode_nnw_thw_mat(mat.unsqueeze(0))
    assert set(spans) == set(decoded[0])


def test_encode_decode_nnw_nsw_thw_mat():
    spans = [[(0, 1, 2), (4, 5)], [(3, 4, 5), (6, 7), (9, 10)]]
