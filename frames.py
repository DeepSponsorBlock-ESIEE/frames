import os
import ffmpeg
import pandas
from datetime import timedelta

def extract_n_frames(in_path: str, out_dir: str, t_range: "tuple[int, int]", t_length: int):
    assert os.path.exists(in_path)
    assert t_range[0] >= 0 and t_range[1] >= t_range[0]

    center = int(sum(t_range)//len(t_range))
    start = int(center - t_length//2)
    end = int(center + t_length//2)

    print(start, end)

    step = 1

    p, _ = os.path.splitext(in_path)
    id = os.path.basename(p)

    for k in range(start, end, step):
        ss = str(timedelta(seconds=k))

        in_file = ffmpeg.input(in_path, ss=ss)
        (
            in_file
            .output(os.path.join(out_dir, f"{id}_{k}.jpg"), vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )

def extract_dfrange_frames(path: str, df: pandas.DataFrame, range_columns: "tuple[str|None, str|None]", t_length: int, out_dir: str):
    assert "videoID" in df.columns
    assert not all([x is None for x in range_columns])
    assert all([x is None or x in df.columns for x in range_columns])

    p, _ = os.path.splitext(path)
    id = os.path.basename(p)

    rows = df[df["videoID"] == id]

    if len(rows) > 0:
        row = rows.iloc[0]
        start = int(row[range_columns[0]]) if range_columns[0] is not None else 1
        end = int(row[range_columns[1]]) if range_columns[1] is not None else float(ffmpeg.probe(path)["format"]["duration"])

        extract_n_frames(path, out_dir, (start, end), t_length)


def extract_sponsored_frames(path: str, df: pandas.DataFrame, t_length: int, out_dir: str):
    extract_dfrange_frames(path, df, ("startTime", "endTime"), t_length, out_dir)

def extract_not_sponsored_frames(path: str, df: pandas.DataFrame, t_length: int, out_dir: str):
    extract_dfrange_frames(path, df, (None, "startNotSure"), t_length//2, out_dir)
    extract_dfrange_frames(path, df, ("endNotSure", None), t_length//2, out_dir)

def extract_frames(path: str, df: pandas.DataFrame, t_length: int, out_dir_sponsored: str, out_dir_not_sponsored: str):
    extract_sponsored_frames(path, df, t_length, out_dir_sponsored)
    extract_not_sponsored_frames(path, df, t_length, out_dir_not_sponsored)