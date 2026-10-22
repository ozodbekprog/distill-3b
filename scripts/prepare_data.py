#!/usr/bin/env python3
"""Parquet -> compact npz shards for KD training (int32 ids, fp16 logits, int32 top-k idx)."""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq

STATS = {"shards": 0, "rows": 0, "seq_lens": set(), "align_hits": 0, "align_total": 0, "skipped": 0}


def process_shard(src, dst_dir):
    out = os.path.join(dst_dir, os.path.basename(src).replace(".parquet", ".npz"))
    if os.path.exists(out):
        return out, None
    pf = pq.ParquetFile(src)
    ids_all, logits_all, idx_all = [], [], []
    align_hits = align_total = 0
    for batch in pf.iter_batches(batch_size=8):
        d = batch.to_pydict()
        for ids, logits, idx in zip(d["input_ids"], d["top_values"], d["token_ids"]):
            ids_a = np.asarray(ids, dtype=np.int32)
            logits_a = np.asarray(logits, dtype=np.float32)
            idx_a = np.asarray(idx, dtype=np.int32)
            if idx_a.shape[0] > 1:
                align_hits += int((idx_a[:-1, 0] == ids_a[1:]).sum())
                align_total += idx_a.shape[0] - 1
            ids_all.append(ids_a)
            logits_all.append(logits_a.astype(np.float16))
            idx_all.append(idx_a)
    lens = {len(x) for x in ids_all}
    if len(lens) == 1:
        ids_o = np.stack(ids_all)
        logits_o = np.stack(logits_all)
        idx_o = np.stack(idx_all)
        mask_o = None
    else:
        L = max(lens)
        n = len(ids_all)
        ids_o = np.zeros((n, L), dtype=np.int32)
        logits_o = np.zeros((n, L, logits_all[0].shape[1]), dtype=np.float16)
        idx_o = np.zeros((n, L, idx_all[0].shape[1]), dtype=np.int32)
        mask_o = np.zeros((n, L), dtype=np.int8)
        for i, (a, b, c) in enumerate(zip(ids_all, logits_all, idx_all)):
            ids_o[i, : len(a)] = a
            logits_o[i, : len(b)] = b
            idx_o[i, : len(c)] = c
            mask_o[i, : len(a)] = 1
    if mask_o is None:
        np.savez(out, ids=ids_o, logits=logits_o, idx=idx_o)
    else:
        np.savez(out, ids=ids_o, logits=logits_o, idx=idx_o, mask=mask_o)
    return out, (lens, align_hits, align_total, len(ids_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/llama405b-logits")
    ap.add_argument("--out", default="data/kd-cache")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "data-shard-*.parquet")))
    print(f"found {len(files)} parquet shards", flush=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, f, args.out): f for f in files}
        for fut in as_completed(futs):
            path, st = fut.result()
            done += 1
            if st is None:
                STATS["skipped"] += 1
            else:
                lens, ah, at, rows = st
                STATS["shards"] += 1
                STATS["rows"] += rows
                STATS["seq_lens"] |= lens
                STATS["align_hits"] += ah
                STATS["align_total"] += at
            if done % 50 == 0:
                el = time.time() - t0
                print(f"{done}/{len(files)} shards, {el:.0f}s, {done/el:.1f} shard/s", flush=True)

    STATS["seq_lens"] = sorted(STATS["seq_lens"])
    STATS["align_frac"] = round(STATS["align_hits"] / max(1, STATS["align_total"]), 4)
    STATS["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(STATS, f, indent=2)
    print(json.dumps(STATS, indent=2))
    print("PREPARE DONE")


if __name__ == "__main__":
    main()
