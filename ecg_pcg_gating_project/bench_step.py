import time

import pandas as pd
import torch

from src.config import load_config
from src.data_loader import load_signal_cache
from src.dataset import WindowDataset
from src.model import build_model
from src.pipeline import windows_path, cache_path
from torch.utils.data import DataLoader

cfg = load_config("config/default.yaml")
df = pd.read_csv(windows_path(cfg, "e1") / "windows.csv")
cache = load_signal_cache(cache_path(cfg))
train = df[df["fold"] != 0].reset_index(drop=True)
ds = WindowDataset(train, cache, cfg, None, None, "pcg", normalize=False)
length = ds.expected_length

crit = torch.nn.CrossEntropyLoss()


def run(batch_size: int, amp: bool, n: int = 30) -> None:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = build_model(cfg, length, "pcg").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    it = iter(loader)
    for _ in range(3):  # warmup
        b = next(it)
        x = b["x"].cuda(); y = b["label"].cuda().long()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        b = next(it)
        x = b["x"].cuda(); y = b["label"].cuda().long()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    steps_per_epoch = len(ds) / batch_size
    print(f"batch {batch_size:4d} amp={amp}: {dt*1000:6.1f} ms/step -> {dt*steps_per_epoch:6.0f} s/epoch")


for bs in (8, 32, 128, 512):
    run(bs, True)
run(8, False)
run(128, False)
