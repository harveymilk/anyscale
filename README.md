# Ray Video Clip Summarizer (Anyscale Job Demo)

This repo shows a *clean, scalable* Ray Data pipeline you can run as an **Anyscale Job**.
It:
1) Loads videos (e.g., from S3) via **Ray Data**  
2) **Chunks** each video into 30‑second clips  
3) **Samples frames** from each clip  
4) Runs a **GPU** model (torchvision ResNet50) on those frames and emits a simple per‑clip summary (top labels)

> ✅ The model choice is intentionally simple—customers can swap in their own (Whisper, BLIP, CLIP, etc.).
> ✅ The code is structured to scale horizontally and saturate multiple GPUs.

---

## Quick Start (locally with CPU/GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Ray (single node)
ray start --head

# Run on a local file or a couple S3 URIs (requires creds in env if S3)
python main.py       --uris "file:///path/to/video1.mp4,file:///path/to/video2.mp4"       --output results.jsonl
```

Each output row contains:
```json
{"video_path": "...", "clip_index": 0, "start_sec": 0.0, "end_sec": 30.0, "summary": "dog, park, person"}
```

---

## Run as an **Anyscale Job**

1. **Create a cluster environment** (one‑time) from `cluster_env.yaml`:
   ```bash
   anyscale cluster-env build --name video-clip-summarizer-env --config cluster_env.yaml
   ```

2. **Launch a cluster** with cheap GPUs (A10G or L40S) using `cluster_compute.yaml`:
   ```bash
   anyscale cluster create --config cluster_compute.yaml
   # Note the printed CLUSTER_ID
   ```

3. **Submit the job** (replace `<CLUSTER_ID>` and the URIs with your bucket):  
   ```bash
   anyscale job submit          --cluster-id <CLUSTER_ID>          --working-dir .          --entrypoint "python main.py --uris 's3://your-bucket/videos/*.mp4' --output results.jsonl --num-gpu-actors 2"
   ```

   - To pass AWS creds, set env vars on the cluster in `cluster_compute.yaml` or your Anyscale Org’s default secrets.
   - You can also pass a file of URIs:
     ```bash
     anyscale job submit            --cluster-id <CLUSTER_ID>            --working-dir .            --entrypoint "python main.py --uri-file s3_uris.txt --output results.jsonl"
     ```

4. **Fetch results** from the job working directory artifacts or write directly to S3 with `--output s3://.../results.jsonl` (if you mount/write S3 from the runtime).

---

## Tuning & Scaling

- **Saturate GPUs**: set `--num-gpu-actors` to your GPU count. Each “actor” is a Ray Data worker that loads the model once on its GPU and processes clip batches.
- **Batching**: use `--batch-size` to control how many clips a GPU worker processes at once.
- **Frame sampling**: use `--frame-every-sec` to control how many frames per clip are scored.
- **I/O Parallelism**: change `--read-parallelism` (number of parallel readers for video bytes).

**Rule of thumb**:
- If GPUs are underutilized, increase `--num-gpu-actors` and/or `--batch-size`.
- If workers stall on decoding, increase `--read-parallelism` or move decode onto beefier CPUs.

---

## File Overview

- `main.py` — Orchestrates the Ray Data pipeline end‑to‑end.
- `src/video_chunker.py` — Safe decoding + 30s chunking + frame sampling (via Decord).
- `src/udf.py` — GPU batch UDF with ResNet50 on CUDA; produces per‑clip summaries.
- `cluster_env.yaml` — Minimal cluster environment (Ray + Torch + Decord).
- `cluster_compute.yaml` — Example cluster spec using A10G/L40S workers.
- `requirements.txt` — For local runs (Anyscale builds from `cluster_env.yaml`).

---

## Notes / Shortcuts

- We avoid ffmpeg by using **Decord** for pure‑Python video decode.
- The “summary” is a top‑k label union across sampled frames. Swap `FrameSummarizer` for your own GPU model (e.g., Whisper, BLIP) without changing the pipeline shape.
- If you only have images: point `--uris` at a set of images; the chunker will treat each as a single 30s “clip”.

---

## Example: Generate a tiny dummy video (optional)

```bash
python tools/make_dummy_video.py --out demo.mp4
python main.py --uris "file://$PWD/demo.mp4" --output results.jsonl
```
