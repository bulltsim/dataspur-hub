"""
Dataset for loading fixed-length frame clips for action recognition.
Expects directory structure:
    root_dir/class_name/video_dir/frame_00001.jpg ...
Where class_name corresponds to label.
"""
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import random
import torch
from torch.utils.data import Dataset
import cv2

class ClipDataset(Dataset):
    def __init__(self, root_dir: str, clip_len: int = 16, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.clip_len = clip_len
        self.transform = transform
        self.samples: List[Tuple[List[Path], int]] = []
        # gather classes
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls in classes:
            class_dir = self.root_dir / cls
            for vid_dir in class_dir.iterdir():
                if not vid_dir.is_dir():
                    continue
                frames = sorted([fp for fp in vid_dir.glob("*.jpg") if fp.is_file()])
                if len(frames) < self.clip_len:
                    continue
                self.samples.append((frames, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frames, label = self.samples[idx]
        # choose random start index
        max_start = len(frames) - self.clip_len
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        imgs = []
        for i in range(start_idx, start_idx + self.clip_len):
            img_path = frames[i]
            img = cv2.imread(str(img_path))
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            imgs.append(img)
        clip = torch.stack(imgs)  # shape (clip_len, C, H, W)
        return clip, label
