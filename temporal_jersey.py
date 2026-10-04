"""Jersey numbers from a sequence of crops of one player: the EfficientNet-B0 + LSTM model from
huggingface.co/Akashpaul123/jersey-number-recognition-temporal (Apache-2.0), in
CV_Models/jersey_Num Models/jersey_model.pth (run_models.py --number-reader temporal).

The model reads 8 whole-player crops (squashed to 128x128) of the same player and predicts the
tens and units digit; a tens digit of 0 means a one-digit number. It has no "no number" output,
so every sequence gets a number: the confidence (tens probability x units probability) is what
keeps weak guesses out of run_models.py's per-track votes.

It was trained on soccer footage with only ten numbers (4, 6, 8, 9, 48, 49, 64, 66, 88, 89), so
its tens digit has only ever been 0, 4, 6 or 8 and its units digit 4, 6, 8 or 9.

Each crop goes through the backbone once; its features are kept per player track, and the LSTM
(cheap) runs on the track's latest SEQUENCE_LENGTH crops. Training sequences were sampled evenly
across a tracklet rather than taken from consecutive frames, so a crop is kept only every
SAMPLE_EVERY frames.
"""

from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

SEQUENCE_LENGTH = 8   # crops per sequence, as trained
MIN_SEQUENCE = 4      # crops a track needs before it is read
SAMPLE_EVERY = 3      # frames between a track's kept crops (8 crops span ~0.8 s at 30 fps)
FORGET_AFTER = 150    # frames without a crop before a track's crops are dropped
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class TemporalJerseyRecognizer(nn.Module):
    """The model, as in its model_architecture.py (layer names must match the checkpoint)."""

    def __init__(self, backbone="efficientnet_b0", feature_dim=256, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.feature_projection = nn.Sequential(nn.Linear(self.backbone.num_features, feature_dim), nn.ReLU(),
                                                nn.Dropout(dropout))
        self.lstm = nn.LSTM(feature_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0, bidirectional=True)
        self.tens_head = nn.Sequential(nn.Linear(2 * lstm_hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 10))
        self.units_head = nn.Sequential(nn.Linear(2 * lstm_hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 10))

    def features(self, x):
        """(N, 3, H, W) crops -> (N, feature_dim) per-crop features."""
        return self.feature_projection(self.backbone(x))

    def classify(self, seq):
        """(B, T, feature_dim) feature sequences -> (tens_logits, units_logits), from the last LSTM step."""
        out, _ = self.lstm(seq)
        return self.tens_head(out[:, -1]), self.units_head(out[:, -1])


class TemporalJerseyReader:
    """Reads jersey numbers per player track from sequences of crops. Call reset() between videos."""

    def __init__(self, path, device):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.input_hw = tuple(cfg.get("img_size", (128, 128)))
        self.model = TemporalJerseyRecognizer(cfg.get("backbone", "efficientnet_b0"), cfg.get("feature_dim", 256),
                                              cfg.get("lstm_hidden", 128), cfg.get("lstm_layers", 2),
                                              cfg.get("dropout", 0.3))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.device = device
        self.model.to(device).eval()
        self.reset()

    def reset(self):
        self.tracks = {}   # key -> (deque of features, frame of the last kept crop)
        self.frame = -1

    def _prepare(self, crops):
        h, w = self.input_hw
        imgs = [cv2.cvtColor(cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA if c.shape[0] > h else cv2.INTER_LINEAR),
                             cv2.COLOR_BGR2RGB) for c in crops]
        x = torch.from_numpy(np.stack(imgs)).float().div_(255)
        x = (x - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
        return x.permute(0, 3, 1, 2).contiguous().to(self.device)

    @torch.no_grad()
    def read(self, crops, keys):
        """Call once per frame with that frame's player crops (BGR, whole player box) and their
        track keys (None for a box without a track, which is read on its own). Returns one
        (number, confidence) or None per crop."""
        self.frame += 1
        out = [None] * len(crops)
        take = [i for i, k in enumerate(keys) if k is None or k not in self.tracks
                or self.frame - self.tracks[k][1] >= SAMPLE_EVERY]
        if not take:
            return out
        feats = self.model.features(self._prepare([crops[i] for i in take]))
        seqs, idx = [], []
        for f, i in zip(feats, take):
            k = keys[i]
            if k is None:
                seqs.append(f[None]); idx.append(i)
                continue
            buf = self.tracks.get(k, (deque(maxlen=SEQUENCE_LENGTH), 0))[0]
            buf.append(f)
            self.tracks[k] = (buf, self.frame)
            if len(buf) >= MIN_SEQUENCE:
                seqs.append(torch.stack(list(buf))); idx.append(i)
        for k in [k for k, (_, last) in self.tracks.items() if self.frame - last > FORGET_AFTER]:
            del self.tracks[k]
        # Sequences of different lengths: group by length so each LSTM call is one batch.
        by_len = {}
        for s, i in zip(seqs, idx):
            by_len.setdefault(len(s), []).append((s, i))
        for group in by_len.values():
            tens, units = self.model.classify(torch.stack([s for s, _ in group]))
            pt, t = tens.softmax(-1).max(-1)
            pu, u = units.softmax(-1).max(-1)
            for (_, i), tt, uu, c in zip(group, t.tolist(), u.tolist(), (pt * pu).tolist()):
                out[i] = (str(tt * 10 + uu), c)
        return out
