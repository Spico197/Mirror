import torch
import torch.nn as nn
from rex.utils.iteration import windowed_queue_iter
from transformers import BertModel

from src.utils import decode_nnw_nsw_thw_mat, decode_nnw_thw_mat, decode_pointer_mat


class Biaffine(nn.Module):
    """Biaffine transformation

    References:
        - https://github.com/yzhangcs/parser/blob/main/supar/modules/affine.py
        - https://github.com/ljynlp/W2NER
    """

    def __init__(self, n_in, n_out=2, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # s = s.permute(0, 2, 3, 1)

        return s


class Triaffine(nn.Module):
    r"""
    Triaffine layer for second-order scoring :cite:`zhang-etal-2020-efficient,wang-etal-2019-second`.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (Optional[int]):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (Optional[float]):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
        decompose (bool):
            If ``True``, represents the weight as the product of 3 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.

    Acknowledge:
        https://github.com/yzhangcs/parser/blob/3c7b20905c90c587a0296338a444ece346b4a4fb/supar/modules/affine.py#L134
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_proj: Optional[int] = None,
        dropout: Optional[float] = 0,
        scale: int = 0,
        bias_x: bool = False,
        bias_y: bool = False,
        decompose: bool = False,
        init: Callable = nn.init.zeros_,
    ) -> Triaffine:
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_proj = n_proj
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        self.init = init

        if n_proj is not None:
            self.mlp_x = MLP(n_in, n_proj, dropout)
            self.mlp_y = MLP(n_in, n_proj, dropout)
            self.mlp_z = MLP(n_in, n_proj, dropout)
        self.n_model = n_proj or n_in
        if not decompose:
            self.weight = nn.Parameter(
                torch.Tensor(
                    n_out, self.n_model + bias_x, self.n_model, self.n_model + bias_y
                )
            )
        else:
            self.weight = nn.ParameterList(
                (
                    nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x)),
                    nn.Parameter(torch.Tensor(n_out, self.n_model)),
                    nn.Parameter(torch.Tensor(n_out, self.n_model + bias_y)),
                )
            )

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.n_proj is not None:
            s += f", n_proj={self.n_proj}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.decompose:
            for i in self.weight:
                self.init(i)
        else:
            self.init(self.weight)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if hasattr(self, "mlp_x"):
            x, y, z = self.mlp_x(x), self.mlp_y(y), self.mlp_z(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len, seq_len]
        if self.decompose:
            wx = torch.einsum("bxi,oi->box", x, self.weight[0])
            wz = torch.einsum("bzk,ok->boz", z, self.weight[1])
            wy = torch.einsum("byj,oj->boy", y, self.weight[2])
            s = torch.einsum("box,boz,boy->bozxy", wx, wz, wy)
        else:
            w = torch.einsum("bzk,oikj->bozij", z, self.weight)
            s = torch.einsum("bxi,bozij,byj->bozxy", x, w, y)
        return s.squeeze(1) / self.n_in**self.scale


class LinearWithAct(nn.Module):
    def __init__(self, n_in, n_out, dropout=0) -> None:
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class PointerMatrix(nn.Module):
    """Pointer Matrix Prediction

    References:
        - https://github.com/ljynlp/W2NER
    """

    def __init__(
        self,
        hidden_size,
        biaffine_size,
        cls_num=2,
        dropout=0,
        biaffine_bias=False,
        use_rope=False,
    ):
        super().__init__()
        self.linear_h = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.linear_t = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.biaffine = Biaffine(
            n_in=biaffine_size,
            n_out=cls_num,
            bias_x=biaffine_bias,
            bias_y=biaffine_bias,
        )
        self.use_rope = use_rope

    def sinusoidal_position_embedding(self, qw, kw):
        batch_size, seq_len, output_dim = qw.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        pos_emb = position_ids * indices
        pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)
        pos_emb = pos_emb.repeat((batch_size, *([1] * len(pos_emb.shape))))
        pos_emb = torch.reshape(pos_emb, (batch_size, seq_len, output_dim))
        pos_emb = pos_emb.to(qw)

        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.cat([-qw[..., 1::2], qw[..., ::2]], -1)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.cat([-kw[..., 1::2], kw[..., ::2]], -1)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def forward(self, x):
        h = self.linear_h(x)
        t = self.linear_t(x)
        if self.use_rope:
            h, t = self.sinusoidal_position_embedding(h, t)
        o = self.biaffine(h, t)
        return o


def multilabel_categorical_crossentropy(y_pred, y_true, bit_mask=None):
    """
    https://kexue.fm/archives/7359
    https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/common/utils.py
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    if bit_mask is None:
        return neg_loss + pos_loss
    else:
        raise NotImplementedError


class MrcPointerMatrixModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        cls_num: int = 2,
        biaffine_size: int = 384,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        # None type id: 0, Next Neighboring Word (NNW): 1, Tail Head Word (THW): 2
        self.none_type_id = none_type_id
        # input: cls instruction sep text sep pad
        # mask:   1       2       3   4    5   0
        self.text_mask_id = text_mask_id

        self.plm = BertModel.from_pretrained(plm_dir)
        hidden_size = self.plm.config.hidden_size
        # self.biaffine_size = biaffine_size
        self.nnw_mat = PointerMatrix(
            hidden_size, biaffine_size, cls_num=2, dropout=dropout
        )
        self.thw_mat = PointerMatrix(
            hidden_size, biaffine_size, cls_num=2, dropout=dropout
        )
        self.criterion = nn.CrossEntropyLoss()

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        mask_mat = (
            mask.eq(self.text_mask_id).unsqueeze(-1).expand((bs, seq_len, seq_len))
        )
        # bit_mask: (batch_size, seq_len, seq_len, 1)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).long()
        )
        return bit_mask

    def forward(self, input_ids, mask, labels=None, is_eval=False, **kwargs):
        hidden = self.input_encoding(input_ids, mask)
        nnw_hidden = self.nnw_mat(hidden)
        thw_hidden = self.thw_mat(hidden)
        # nnw_hidden = nnw_hidden / self.biaffine_size ** 0.5
        # thw_hidden = thw_hidden / self.biaffine_size ** 0.5
        # # (bs, 2, seq_len, seq_len)
        bs, _, seq_len, seq_len = nnw_hidden.shape

        bit_mask = self.build_bit_mask(mask)

        results = {"logits": {"nnw": nnw_hidden, "thw": thw_hidden}}
        if labels is not None:
            # mean
            nnw_loss = self.criterion(
                nnw_hidden.permute(0, 2, 3, 1).reshape(-1, 2),
                labels[:, 0, :, :].reshape(-1),
            )
            thw_loss = self.criterion(
                thw_hidden.permute(0, 2, 3, 1).reshape(-1, 2),
                labels[:, 1, :, :].reshape(-1),
            )
            loss = nnw_loss + thw_loss
            results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(nnw_hidden, thw_hidden, bit_mask, **kwargs)
            results["pred"] = batch_positions
        return results

    def decode(
        self,
        nnw_hidden: torch.Tensor,
        thw_hidden: torch.Tensor,
        bit_mask: torch.Tensor,
        **kwargs,
    ):
        # B x L x L
        nnw_pred = nnw_hidden.argmax(1)
        thw_pred = thw_hidden.argmax(1)
        # B x 2 x L x L
        pred = torch.stack([nnw_pred, thw_pred], dim=1)
        pred = pred * bit_mask

        batch_preds = decode_nnw_thw_mat(pred, offsets=kwargs.get("offset"))

        return batch_preds


class MrcGlobalPointerModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        use_rope: bool = True,
        cls_num: int = 2,
        biaffine_size: int = 384,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: float = 0.3,
        mode: str = "w2",
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        # None type id: 0, Next Neighboring Word (NNW): 1, Tail Head Word (THW): 2
        self.none_type_id = none_type_id
        # input: cls instruction sep text sep pad
        # mask:   1       2       3   4    5   0
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        # mode: w2: w2ner, cons: consecutive spans
        self.mode = mode
        assert self.mode in ["w2", "cons"]

        self.plm = BertModel.from_pretrained(plm_dir)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=2 if self.mode == "w2" else 1,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        mask_mat = (
            mask.eq(self.text_mask_id).unsqueeze(-1).expand((bs, seq_len, seq_len))
        )
        # bit_mask: (batch_size, 1, seq_len, seq_len)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        )
        if self.mode == "cons":
            bit_mask = bit_mask.triu()

        return bit_mask

    def forward(
        self, input_ids, mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        # (bs, 2, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 2, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        assert self.mode in ["w2", "cons"]
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            if self.mode == "w2":
                for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                    path_prob *= probs[0, se[0], se[-1]]
                path_prob *= probs[1, path[-1], path[0]]
            elif self.mode == "cons":
                path_prob = probs[0, path[0], path[-1]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        **kwargs,
    ):
        # mode: w2: w2ner with nnw and thw labels, cons: consecutive spans with one type of labels
        assert self.mode in ["w2", "cons"]
        # B x 2 x L x L
        probs = logits.sigmoid()
        pred = (probs > top_p).long()
        if self.mode == "w2":
            preds = decode_nnw_thw_mat(pred, offsets=kwargs.get("offset"))
        elif self.mode == "cons":
            pred = pred.triu()
            preds = decode_pointer_mat(pred, offsets=kwargs.get("offset"))

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        return batch_preds


class SchemaGuidedInstructBertModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        use_rope: bool = True,
        biaffine_size: int = 512,
        none_type_id: int = 0,
        label_mask_id: int = 3,
        text_mask_id: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        # None type id: 0, Next Neighboring Word (NNW): 1, Next Span Word (NSW): 2, Tail Head Word (THW): 3
        self.none_type_id = none_type_id
        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  0     1   2           3    4   3    4   3    4   5    6    7   8          9     0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = BertModel.from_pretrained(plm_dir)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # bit_mask: (batch_size, 1, seq_len, seq_len)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        )

        return bit_mask

    def forward(
        self, input_ids, mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        **kwargs,
    ):
        # B x 3 x L x L
        probs = logits.sigmoid()
        pred = (probs > top_p).long()
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        return batch_preds
