# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
import torch.nn as nn
from typing import List, Optional, Callable, Union


class Embedding(nn.Module):
    # TODO: check if it is possible to train this on a larger dataset?

    def __init__(self, n_frames, n_dims, height=1, width=1, std=0.1, **kwargs):
        super().__init__()
        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.n_dims = n_dims
        self.frame_embs = nn.Embedding(n_frames, height * width * n_dims, max_norm=float(n_dims))
        th.nn.init.normal_(self.frame_embs.weight.data, 0.0, std)

    def average(self):
        return self.frame_embs.weight.mean(0, keepdims=True)

    def median(self):
        return self.frame_embs.weight.median(0, keepdims=True)

    def forward(self, idxs):
        bn = idxs.shape[0]
        if idxs.max() >= self.n_frames:
            idxs = idxs.clamp(max=self.n_frames-1)
        embs = self.frame_embs(idxs)
        return embs.reshape(bn, self.n_dims, self.height, self.width)


class TemporalEmbedding(nn.Module):
    def __init__(self, n_frames, height, width, n_dims, ksize=5, std=0.1, temp=5.0):
        super().__init__()
        assert ksize % 2 == 1
        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.n_dims = n_dims
        self.ksize = ksize
        self.pad = ksize // 2
        # TODO: what should be max norm?
        self.frame_embs = nn.Embedding(n_frames + self.pad * 2, height * width * n_dims, max_norm=float(n_dims))
        th.nn.init.normal_(self.frame_embs.weight.data, 0.0, std)
        weights = th.exp((-((th.arange(self.ksize) - self.pad) ** 2)) / temp)
        weights = weights / th.sum(weights)
        self.register_buffer("weights", weights)

    def average(self):
        return self.frame_embs.weight.mean(0, keepdims=True)

    def forward(self, idxs):
        B = idxs.shape[0]
        device = idxs.device
        idxs_nbs = idxs[:, np.newaxis].expand(-1, self.ksize) + th.arange(self.ksize, device=device)
        embs_nbs = self.frame_embs(idxs_nbs.reshape(-1)).reshape(B, self.ksize, -1)
        embs = (embs_nbs * self.weights[np.newaxis, :, np.newaxis]).sum(dim=1)
        return embs.reshape(B, self.n_dims, self.height, self.width)


class CpuEmbedding(nn.Module):
    # TODO: check if it is possible to train this on a larger dataset?

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.frame_embs = nn.Embedding(*args, **kwargs)

    def to(self, *args, **kwargs):
        # NOTE: just ignoring the conversion
        return self

    def forward(self, idxs):
        return self.frame_embs(idxs.cpu()).to(idxs)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        input_dims: int,
        include_input: bool = True,
        num_freqs: int = 8,
        log_sampling: bool = True,
        periodic_fns: List[Callable] = [th.sin, th.cos],
        **kwargs,
    ):
        super(PositionalEncoding, self).__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.kwargs = kwargs
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        fn_names = []
        fn_scales = []
        freq_pows = []
        d = self.input_dims

        out_dim = 0
        if self.include_input:
            # embed_fns.append(lambda x, **kwargs: x)
            # freq_pows.append(-1)
            fn_scales.append(0.0)
            out_dim += d

        max_freq = self.num_freqs - 1
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** th.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = th.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            freq_pow = th.log2(freq)
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq.item()))
                fn_names.append(p_fn.__name__)
                freq_pows.append(freq_pow)
                fn_scales.append(freq.item())

                out_dim += d

        self.freq_bands = freq_bands
        self.embed_fns = embed_fns
        self.fn_names = fn_names
        self.out_dims = out_dim
        self.freq_k = th.tensor(freq_pows).reshape(1, 1, -1, 1)
        self.fn_scales = fn_scales

    @property
    def dims(self):
        return self.out_dims

    def forward(self, inputs: th.Tensor, weights: Optional[th.Tensor] = None, **kwargs):
        return self._embed(inputs, weights=weights, **kwargs)

    def _embed(self, inputs: th.Tensor, weights: Optional[th.Tensor] = None, **kwargs):
        if self.num_freqs == 0:
            assert self.include_input
            return inputs, None

        # embedded_ =  th.cat([fn(inputs) for fn in self.embed_fns], -1)
        inputs_expand = inputs[..., None, :]
        n_dims = inputs_expand.dim()
        freq_bands = self.freq_bands.to(inputs.device).reshape((1,) * (n_dims - 2) + (-1, 1))
        inputs_bands = inputs_expand * freq_bands
        sin_component = th.sin(inputs_bands)
        cos_component = th.cos(inputs_bands)
        embedded = th.stack([sin_component, cos_component], dim=-2).flatten(start_dim=-3)
        # assert th.allclose(embedded_, embedded, atol=1e-6)

        if weights is not None:
            embedded = embedded * weights
        if self.include_input:
            embedded = th.cat([inputs, embedded], -1)
        return embedded, None

    def update_threshold(self, *args, **kwargs):
        pass

    def update_tau(self, *args, **kwargs):
        pass

    def get_tau(self):
        return 0.0


class Optcodes(nn.Module):
    def __init__(
        self,
        n_codes: int,
        code_ch: int,
        idx_map: Optional[Union[np.ndarray, th.Tensor]] = None,
        transform_code: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.n_codes = n_codes
        self.code_ch = code_ch
        self.codes = nn.Embedding(n_codes, code_ch)
        self.transform_code = transform_code
        self.idx_map = None
        if idx_map is not None:
            self.idx_map = th.LongTensor(idx_map)
        self.init_parameters(mean, std)

    def forward(self, idx: th.Tensor, t: Optional[th.Tensor] = None, *args, **kwargs):
        shape = idx.shape[:-1]
        if self.idx_map is not None:
            idx = self.idx_map[idx.long()].to(idx.device)
        if not self.training and idx.max() < 0:
            codes = self.codes.weight.mean(0, keepdims=True).expand(len(idx), -1)
        else:
            if idx.shape[-1] != 1:
                codes = self.codes(idx[..., :2].long()).squeeze(1)
                w = idx[..., 2]
                # interpolate given mixing weights
                codes = th.lerp(codes[..., 0, :], codes[..., 1, :], w[..., None])
            else:
                if idx.max() > self.n_codes:
                    idx = idx.clamp(max=self.n_codes - 1)
                    print('Warning! Out-of-range index detected in Optcodes input. Clamp it to self.n_codes-1')
                    print('Check the code if this is not expected')
                codes = self.codes(idx.long()).squeeze(1)

        if self.transform_code:
            codes = codes.view(t.shape[0], 4, -1).flatten(start_dim=-2)
        return codes

    def init_parameters(self, mean: Optional[float] = None, std: Optional[float] = None):
        if mean is None:
            nn.init.xavier_normal_(self.codes.weight)
            return

        if std is not None and std > 0.0:
            nn.init.normal_(self.codes.weight, mean=mean, std=std)
        else:
            nn.init.constant_(self.codes.weight, mean)
