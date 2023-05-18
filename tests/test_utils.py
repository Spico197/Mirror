import torch

from src.utils import (
    decode_nnw_nsw_thw_mat,
    decode_nnw_thw_mat,
    encode_nnw_nsw_thw_mat,
    encode_nnw_thw_mat,
    find_paths_from_adj_mat,
)


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
    spans = [[(18,), (39,)], [(22,), (35, 37)], [(24,), (46,)], [(24,), (46, 49)]]
    spans = sorted(spans)
    labels = encode_nnw_nsw_thw_mat(spans, 50)
    # nnw
    assert labels[0, 18, 39].item() == 1
    assert labels[0, 22, 35].item() == 1
    assert labels[0, 35, 37].item() == 1
    assert labels[0, 24, 46].item() == 1
    assert labels[0, 46, 49].item() == 1
    # nsw
    assert labels[1, 18, 39].item() == 1
    assert labels[1, 22, 35].item() == 1
    assert labels[1, 24, 46].item() == 1
    # thw
    assert labels[2, 39, 18].item() == 1
    assert labels[2, 37, 22].item() == 1
    assert labels[2, 46, 24].item() == 1
    assert labels[2, 49, 24].item() == 1
    # total
    assert labels.sum().item() == 12
    decoded = decode_nnw_nsw_thw_mat(labels.unsqueeze(0))[0]
    decoded = sorted(decoded)
    assert decoded == spans


def test_recursive_encode_decode_nnw_nsw_thw_mat():
    spans = [
        [(51,), (117,), (115,)],
        [(29,), (117,), (115,)],
        [(42,), (117,), (115,)],
        [(40,), (115,), (117,)],
    ]
    spans = sorted(spans)
    labels = encode_nnw_nsw_thw_mat(spans, 120)
    decoded = decode_nnw_nsw_thw_mat(labels.unsqueeze(0))[0]
    decoded = sorted(decoded)
    assert decoded == spans
