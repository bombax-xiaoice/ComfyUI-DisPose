import random
from typing import List
from einops import rearrange, repeat

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import pdb
import time

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.SiLU(inplace=False),
            nn.Linear(mid_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.mlp(x)

def vectorized_bilinear_interpolation(level_adapter_state, coords, frame_idx, interpolated_values):
    x = coords[:, 0]
    y = coords[:, 1]

    x1 = x.floor().long()
    y1 = y.floor().long()
    x2 = x1 + 1
    y2 = y1 + 1

    x1 = torch.clamp(x1, 0, level_adapter_state.shape[3] - 1)
    y1 = torch.clamp(y1, 0, level_adapter_state.shape[2] - 1)
    x2 = torch.clamp(x2, 0, level_adapter_state.shape[3] - 1)
    y2 = torch.clamp(y2, 0, level_adapter_state.shape[2] - 1)

    x_frac = x - x1.float()
    y_frac = y - y1.float()

    w11 = (1 - x_frac) * (1 - y_frac)
    w21 = x_frac * (1 - y_frac)
    w12 = (1 - x_frac) * y_frac
    w22 = x_frac * y_frac

    for i, (x1_val, y1_val, x2_val, y2_val, w11_val, w21_val, w12_val, w22_val, interpolated_value) in enumerate(zip(x1, y1, x2, y2, w11, w21, w12, w22, interpolated_values)):
        level_adapter_state[frame_idx, :, y1_val, x1_val] += interpolated_value * w11_val
        level_adapter_state[frame_idx, :, y1_val, x2_val] += interpolated_value * w21_val
        level_adapter_state[frame_idx, :, y2_val, x1_val] += interpolated_value * w12_val
        level_adapter_state[frame_idx, :, y2_val, x2_val] += interpolated_value * w22_val

    return level_adapter_state

def bilinear_interpolation(level_adapter_state, x, y, frame_idx, interpolated_value):
    # note the boundary
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1
    x_frac = x - x1
    y_frac = y - y1

    x1, x2 = max(min(x1, level_adapter_state.shape[3] - 1), 0), max(min(x2, level_adapter_state.shape[3] - 1), 0)
    y1, y2 = max(min(y1, level_adapter_state.shape[2] - 1), 0), max(min(y2, level_adapter_state.shape[2] - 1), 0)

    w11 = (1 - x_frac) * (1 - y_frac)
    w21 = x_frac * (1 - y_frac)
    w12 = (1 - x_frac) * y_frac
    w22 = x_frac * y_frac

    level_adapter_state[frame_idx, :, y1, x1] += interpolated_value * w11
    level_adapter_state[frame_idx, :, y1, x2] += interpolated_value * w21
    level_adapter_state[frame_idx, :, y2, x1] += interpolated_value * w12
    level_adapter_state[frame_idx, :, y2, x2] += interpolated_value * w22

    return level_adapter_state

class PointAdapter(nn.Module):

    def __init__(
        self,
        embedding_channels=1280,
        channels=[320, 640, 1280, 1280],
        downsample_rate=[16, 32, 64, 64],
        mid_dim=128
    ):
        super().__init__()

        self.model_list = nn.ModuleList()

        for ch in channels:
            self.model_list.append(MLP(embedding_channels, ch, mid_dim))

        self.downsample_rate = downsample_rate
        self.embedding_channels = embedding_channels
        self.channels = channels
        self.radius = 4

    def generate_loss_mask(self, batch_size, point_tracker, num_frames, h, w, loss_type):
        downsample_rate = self.downsample_rate[0]
        level_w, level_h = w // downsample_rate, h // downsample_rate
        if loss_type == 'global':
            loss_mask = torch.ones((batch_size, num_frames, 4, level_h, level_w))
        else:
            loss_mask = torch.zeros((batch_size, num_frames, 4, level_h, level_w))
            for batch_idx in range(batch_size):
                for frame_idx in range(num_frames):
                    if self.training:
                        keypoints, subsets = point_tracker[frame_idx]["candidate"][batch_idx], point_tracker[frame_idx]["subset"][batch_idx][0]
                    else:
                        keypoints, subsets = point_tracker[frame_idx]["candidate"], point_tracker[frame_idx]["subset"][0]
                        assert batch_size == 1
                    for point_idx, (keypoint, subset) in enumerate(zip(keypoints, subsets)):
                        if subset != -1:
                            px, py = keypoint[0] * level_w, keypoint[1] * level_h

                            x1 = int(px) - self.radius
                            y1 = int(py) - self.radius
                            x2 = int(px) + self.radius
                            y2 = int(py) + self.radius

                            x1, x2 = max(min(x1, level_w - 1), 0), max(min(x2, level_w - 1), 0)
                            y1, y2 = max(min(y1, level_h - 1), 0), max(min(y2, level_h - 1), 0)
                            loss_mask[batch_idx][frame_idx][:, y1:y2, x1:x2] = 1.0

        return loss_mask

    def forward(self, point_tracker, size, point_embedding, pose_latents, index_list=None, drop_rate=0.0, loss_type='global') -> List[torch.Tensor]:
        w, h = size
        num_frames = len(point_tracker)
        batch_size, num_points, _ = point_embedding.shape

        loss_mask = self.generate_loss_mask(batch_size, point_tracker, num_frames, h, w, loss_type)

        downsample_rate = self.downsample_rate[0]
        level_w, level_h = w // downsample_rate, h // downsample_rate
        level_adapter_state = torch.zeros((batch_size, num_frames, self.embedding_channels, level_h, level_w)).to(point_embedding.device, dtype=point_embedding.dtype)
        level_mask = torch.zeros((batch_size, num_frames, level_h, level_w)).to(point_embedding.device, dtype=point_embedding.dtype)
        level_count = torch.ones((batch_size, num_frames, level_h, level_w)).to(point_embedding.device, dtype=point_embedding.dtype)
        for batch_idx in range(batch_size):
            for frame_idx in range(num_frames):
                if self.training:
                    keypoints, subsets = point_tracker[frame_idx]["candidate"][batch_idx], point_tracker[frame_idx]["subset"][batch_idx][0]
                else:
                    keypoints, subsets = point_tracker[frame_idx]["candidate"], point_tracker[frame_idx]["subset"][0]
                    assert batch_size == 1
                for point_idx, (keypoint, subset) in enumerate(zip(keypoints, subsets)):
                    if keypoint.min() < 0:
                        continue
                    px, py = keypoint[0] * level_w, keypoint[1] * level_h
                    px, py = max(min(int(px), level_w - 1), 0), max(min(int(py), level_h - 1), 0)
                    if subset != -1:
                        if point_embedding[batch_idx, point_idx].mean() != 0 or random.random() > drop_rate:
                            if level_mask[batch_idx, frame_idx, py, px] !=0:
                                level_count[batch_idx, frame_idx, py, px] +=1
                            level_adapter_state[batch_idx, frame_idx, :, py, px] += point_embedding[batch_idx, point_idx]
                            level_mask[batch_idx, frame_idx, py, px] = 1.0
        
        adapter_state = []
        level_adapter_state = level_adapter_state/level_count.unsqueeze(2)
        level_adapter_state = rearrange(level_adapter_state, "b f c h w-> b f h w c")
        for level_idx, module in enumerate(self.model_list):
            downsample_rate = self.downsample_rate[level_idx]
            level_w, level_h = w // downsample_rate, h // downsample_rate

            point_feat = module(level_adapter_state)
            point_feat = point_feat * level_mask.unsqueeze(-1)

            point_feat = rearrange(point_feat, "b f h w c-> (b f) c h w")
            point_feat = nn.Upsample(size=(level_h, level_w), mode='bilinear')(point_feat)

            temp_mask = rearrange(level_mask, "b f h w-> (b f) h w")
            temp_mask = nn.Upsample(size=(level_h, level_w), mode='nearest')(temp_mask.unsqueeze(1))
            point_feat = point_feat * temp_mask

            point_feat = rearrange(point_feat, "(b f) c h w-> b f c h w", b=batch_size)
            adapter_state.append(point_feat)
        
        return adapter_state, loss_mask
