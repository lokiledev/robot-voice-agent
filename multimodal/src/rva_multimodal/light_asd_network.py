"""Light-ASD's network, in PyTorch, to convert its checkpoint to ONNX.

From https://github.com/Junhua-Liao/Light-ASD (model/Encoder.py, Classifier.py, Model.py, loss.py)
at ed38c232, MIT License, Copyright (c) 2023 Liao Junhua. Changed: the audio encoder pools with
MaxPool2d where the original runs MaxPool3d on 4D tensors, which PyTorch reads as unbatched and so
pools the same way, but which ONNX cannot export; the weight initialization, unused once the
checkpoint is loaded, is left out; the layers keep their names, which the checkpoint's keys use.
"""

import torch
from torch import nn


class _AudioBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.m_3 = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.m_5 = nn.Conv2d(in_channels, out_channels, (5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(out_channels, out_channels, (1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.last = nn.Conv2d(out_channels, out_channels, (1, 1), bias=False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_m_3(self.m_3(x))))))
        x_5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_m_5(self.m_5(x))))))
        return self.relu(self.bn_last(self.last(x_3 + x_5)))


class _VisualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, down: bool) -> None:
        super().__init__()
        stride = (1, 2, 2) if down else (1, 1, 1)
        self.relu = nn.ReLU()
        self.s_3 = nn.Conv3d(
            in_channels, out_channels, (1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False
        )
        self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv3d(out_channels, out_channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.s_5 = nn.Conv3d(
            in_channels, out_channels, (1, 5, 5), stride=stride, padding=(0, 2, 2), bias=False
        )
        self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv3d(out_channels, out_channels, (5, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.last = nn.Conv3d(out_channels, out_channels, (1, 1, 1), bias=False)
        self.bn_last = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_s_3(self.s_3(x))))))
        x_5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_s_5(self.s_5(x))))))
        return self.relu(self.bn_last(self.last(x_3 + x_5)))


class _VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _VisualBlock(1, 32, down=True)
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _VisualBlock(32, 64, down=False)
        self.pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _VisualBlock(64, 128, down=False)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
        x = x.transpose(1, 2)
        batch, frames, channels, height, width = x.shape
        x = self.maxpool(x.reshape(batch * frames, channels, height, width))
        return x.view(batch, frames, channels)


class _AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _AudioBlock(1, 32)
        self.pool1 = nn.MaxPool2d((1, 3), stride=(1, 2), padding=(0, 1))
        self.block2 = _AudioBlock(32, 64)
        self.pool2 = nn.MaxPool2d((1, 3), stride=(1, 2), padding=(0, 1))
        self.block3 = _AudioBlock(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
        return torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)


class _BidirectionalGru(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gru_forward = nn.GRU(channels, channels, batch_first=True)
        self.gru_backward = nn.GRU(channels, channels, batch_first=True)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.gru_forward(x)[0])
        x = torch.flip(self.gru_backward(torch.flip(x, dims=[1]))[0], dims=[1])
        return self.gelu(x)


class _AsdModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visualEncoder = _VisualEncoder()  # The checkpoint's key.
        self.audioEncoder = _AudioEncoder()  # The checkpoint's key.
        self.GRU = _BidirectionalGru(128)  # The checkpoint's key.


class LightAsdNetwork(nn.Module):
    """MFCC (1, 4T, 13) and grayscale faces (1, T, 112, 112) to a speaking logit per frame (T)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _AsdModel()
        self.FC = nn.Linear(128, 2)  # The checkpoint's key, under lossAV.

    def forward(self, audio: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        sound = self.model.audioEncoder(audio.unsqueeze(1).transpose(2, 3))
        batch, frames, height, width = faces.shape
        faces = (faces.view(batch, 1, frames, height, width) / 255 - 0.4161) / 0.1688
        sight = self.model.visualEncoder(faces)
        fused = self.model.GRU(sound + sight).reshape(-1, 128)
        return self.FC(fused)[:, 1]

    def load_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        """Load Light-ASD's checkpoint, whose visual-only head is left out."""
        self.load_state_dict(
            {
                key.replace("lossAV.FC.", "FC."): value
                for key, value in state.items()
                if key.startswith(("model.", "lossAV."))
            }
        )
