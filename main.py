import argparse
import os
from typing import List, Optional

import s3fs
import dotenv
import ray
import ray.data as rds
from ray.data import ActorPoolStrategy

from src.video_chunker import explode_into_clips
from src.udf import FrameSummarizer


dotenv.load_dotenv()

def parse_args():
    p = argparse.ArgumentParser(description="Ray Data video chunking + GPU summarization")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--uris", type=str, help="Comma-separated list or glob (e.g., s3://bucket/prefix/*.mp4)")
    g.add_argument("--uri-file", type=str, help="Text file with one URI per line")

    p.add_argument("--output", type=str, default="results.jsonl", help="Local or S3 path (if configured)")
    p.add_argument("--clip-seconds", type=int, default=30)
    p.add_argument("--frame-every-sec", type=float, default=5.0, help="Sample frame every N seconds in a clip")
    p.add_argument("--read-parallelism", type=int, default=8, help="Parallel file readers")
    p.add_argument("--num-gpu-actors", type=int, default=0, help="# of GPU workers to load the model")
    p.add_argument("--batch-size", type=int, default=8, help="Clips per GPU batch")
    p.add_argument("--topk", type=int, default=3, help="Top-k labels per clip summary")
    return p.parse_args()

def parse_uri_list(uris_arg: Optional[str], uri_file: Optional[str]) -> List[str]:
    if uri_file:
        with open(uri_file, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    assert uris_arg is not None
    if "," in uris_arg:
        return [u.strip() for u in uris_arg.split(",") if u.strip()]
    return [uris_arg]


if __name__ == "__main__":
    args = parse_args()

    if not ray.is_initialized():
        ray.init(address="auto")

    uris = parse_uri_list(args.uris, args.uri_file)
    print(f"Reading {len(uris)} URI(s) ...")


    key = os.getenv("APP_AWS_ACCESS_KEY_ID") 
    secret = os.getenv("APP_AWS_SECRET_ACCESS_KEY")
    region = os.getenv("APP_AWS_REGION") or os.getenv("APP_AWS_DEFAULT_REGION") 

    client_kwargs = {"region_name": region} if region else {}
    fs_in = s3fs.S3FileSystem(key=key, secret=secret, client_kwargs=client_kwargs)

    ds = rds.read_binary_files(
        uris,
        include_paths=True,
        override_num_blocks=args.read_parallelism,
        filesystem=fs_in,
    )

    clip_ds = ds.flat_map(
        lambda row: explode_into_clips(
            row,
            clip_seconds=args.clip_seconds,
            frame_every_sec=args.frame_every_sec,
        )
    )

    num_gpus_env = int(os.environ.get("N_GPUS", "0"))
    use_gpu = (args.num_gpu_actors > 0) or (num_gpus_env > 0)

    actor_pool_size = args.num_gpu_actors or (num_gpus_env if num_gpus_env > 0 else 1)

    summarizer = FrameSummarizer  # constructed on workers

    summarized = clip_ds.map_batches(
        summarizer,
        batch_size=args.batch_size,
        batch_format="pandas",
        compute=ActorPoolStrategy(size=actor_pool_size),
        num_gpus=1 if use_gpu else 0,
        fn_constructor_kwargs={"topk": args.topk},
    )

    wanted_cols = [c for c in summarized.schema().names if c not in {"bytes"}]
    summarized = summarized.select_columns(wanted_cols)


    out_path = args.output
    fs_out = make_s3fs_if_needed(out_path)
    print(f"Writing results to {out_path}")

    if out_path.lower().endswith(".jsonl") or out_path.lower().endswith(".json"):
        summarized.write_json(out_path, filesystem=fs_out)
    else:
        summarized.write_parquet(out_path, filesystem=fs_out)

    print("Done.")
