import os
import ffmpeg
import numpy as np
import pandas
from matplotlib import pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video, write_jpeg


def get_t_ranges(df: pd.DataFrame, idx: int):
    assert all([x in df.columns for x in ["videoID", "startTime",
               "endTime", "startNotSure", "endNotSure"]])

    if idx in df.index:
        data = df.iloc[idx]

        return {
            "sponsored": (int(data["startTime"]), int(data["endTime"])),
            "not_sponsored": (0, int(data["startNotSure"]))
        }


def get_images(in_path, t_range, width, height, length=10, fps="2/1"):
    assert os.path.exists(in_path)

    out, err = (
        ffmpeg.input(in_path, ss=t_range[0], to=t_range[1])
        .filter("scale", width, height)
        .filter("fps", fps=fps)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=length)
        .run(capture_stdout=True, quiet=True)
    )

    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return np.moveaxis(frames, -1, 1)  # Move channel axis to first dimension


class DSBDataset(Dataset):
    def __init__(self, infos_df, video_paths, seq_length, img_path, transform=None, target_transform=None):
        self.infos_df = infos_df
        self.video_paths = video_paths
        self.seq_length = seq_length
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = img_path

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        p, _ = os.path.splitext(self.video_paths[index // 2])
        id = os.path.basename(p)

        df_idx = self.infos_df.index[df["videoID"] == id].tolist()[0]
        t_range = get_t_ranges(self.infos_df, df_idx)

        item_class = "sponsored" if index % 2 == 0 else "not_sponsored"
        cache_path = f"{self.img_path}/cache/{item_class}/{id}/{id}_{self.seq_length}.npy"

        # Caching
        if os.path.exists(cache_path):
            extracted_images = np.load(cache_path)
        else:
            os.makedirs(os.path.split(cache_path)[0])
            extracted_images = get_images(
                self.video_paths[index // 2],
                (t_range[item_class][0], t_range[item_class][1]),
                640, 360
            )
            np.save(cache_path, extracted_images)

        label = index % 2

        if self.transform:
            extracted_images = [self.transform(x) for x in extracted_images]
        if self.target_transform:
            label = self.target_transform(label)

        tensor = torch.tensor(extracted_images)

        return tensor, label
