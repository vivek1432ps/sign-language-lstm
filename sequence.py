"""Data, preprocessing and network for an isolated Auslan benchmark."""
import hashlib
import io
from pathlib import Path
import csv
import re
import tarfile
import urllib.request
import zipfile

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "auslan.zip"
URL = "https://archive.ics.uci.edu/static/public/114/australian+sign+language+signs.zip"
SHA = "2939fa3284e2aafdf15054691d441f07e06b7ebadf1c68c3ed662aa039b3d7df"
CLASSES = ("come", "drink", "eat", "go", "hello", "please", "sorry", "thank", "wait(notyet)", "write")
SIGNERS = ("adam", "andrew", "john", "stephen", "waleed")
STEPS, CHANNELS = 48, 9
HELD = "john"


def load_archive(path=CACHE):
    """Read the pinned official archive without extracting attacker-controlled paths."""
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=45) as response:
            blob = response.read(4_000_000)
        if hashlib.sha256(blob).hexdigest() != SHA:
            raise ValueError("Unexpected UCI archive checksum.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != SHA:
        raise ValueError("Unexpected UCI archive checksum.")
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        inner = outer.read("allsigns.tar.gz")
    x, y, signer, paths = [], [], [], []
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            match = re.fullmatch(r"signs/([a-z]+)\d+/([^/]+?)\d+\.sign", member.name)
            if not match or not member.isfile():
                continue
            person, label = match.groups()
            if person not in SIGNERS or label not in CLASSES:
                continue
            with archive.extractfile(member) as file:
                x.append(resample(file.read().decode("ascii")))
            paths.append(member.name)
            signer.append(person)
            y.append(CLASSES.index(label))
    return np.stack(x), np.asarray(y), np.asarray(signer), paths


def resample(data):
    rows = [[float(row[i]) for i in (0,1,2,3,6,7,8,9)]
            for row in csv.reader(io.StringIO(data)) if row]
    v = np.asarray(rows, dtype=np.float32)
    if len(v) < 2 or not np.isfinite(v).all():
        raise ValueError("Expected at least two finite frames.")
    # Roll is periodic. Zero and one describe the same orientation.
    a = v[:, 3] * 2*np.pi
    v = np.column_stack((v[:, :3], np.sin(a), np.cos(a), v[:, 4:]))
    grid = np.linspace(0, 1, len(v))
    target = np.linspace(0, 1, STEPS)
    return np.column_stack([np.interp(target, grid, c) for c in v.T]).astype(np.float32)


class SignLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(CHANNELS, 40, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(80), nn.Dropout(.20), nn.Linear(80, len(CLASSES)))

    def forward(self, x):
        states, _ = self.encoder(x)
        return self.head(states.mean(1))


def predict(model, x, mean, scale):
    with torch.inference_mode():
        return model(torch.as_tensor((x-mean)/scale)).softmax(-1).numpy()
