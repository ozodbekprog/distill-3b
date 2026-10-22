#!/usr/bin/env python3
"""KD trainer: Llama-405B top-50 logits -> Llama-3.2-3B student (Unsloth QLoRA)."""

import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    p.add_argument("--data", default="data/kd-cache")
    p.add_argument("--out", default="models/student-3b-kd")
    p.add_argument("--log", default="logs/kd_train.csv")
    p.add_argument("--backend", default="auto", choices=["auto", "unsloth", "peft"])
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-shard-stride", type=int, default=30)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def shard_rows(path, max_len):
    z = np.load(path)
    n, L = z["ids"].shape
    return n, min(L, max_len)


def load_student(args):
    if args.backend in ("auto", "unsloth"):
        try:
            import unsloth  # noqa: F401
            from unsloth import FastLanguageModel

            model, tok = FastLanguageModel.from_pretrained(
                args.model, max_seq_length=args.max_len, load_in_4bit=True, dtype=None
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=TARGET_MODULES,
                lora_alpha=args.lora_r,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
            if hasattr(FastLanguageModel, "for_training"):
                FastLanguageModel.for_training(model)
            print("backend: unsloth")
            return model, tok, "unsloth"
        except Exception as e:
            if args.backend == "unsloth":
                raise
            print("unsloth unavailable, peft fallback. reason:", repr(e))
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES),
    )
    print("backend: peft")
    return model, tok, "peft"


def compute_loss(model, ids, mask, teacher_logits, teacher_idx, alpha, temp):
    out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    logits = out.logits
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    logp_s = logits.gather(-1, teacher_idx) - lse
    p_t = torch.softmax(teacher_logits / temp, dim=-1)
    logp_t = torch.log(p_t + 1e-9)
    kl = (p_t * (logp_t - logp_s)).sum(-1).mean()
    loss = kl
    ce = torch.tensor(0.0, device=kl.device)
    if alpha > 0:
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), teacher_idx[:, :, 0].reshape(-1))
        loss = alpha * ce + (1 - alpha) * kl
    return loss, kl.detach(), ce.detach(), logits


def batch_from_shard(z, rows, max_len, device):
    ids = torch.from_numpy(z["ids"][rows, :max_len].astype(np.int64)).to(device)
    tlog = torch.from_numpy(z["logits"][rows, :max_len].astype(np.float32)).to(device)
    tidx = torch.from_numpy(z["idx"][rows, :max_len].astype(np.int64)).to(device)
    mask = None
    if "mask" in z.files:
        mask = torch.from_numpy(z["mask"][rows, :max_len].astype(np.int64)).to(device)
    return ids, mask, tlog, tidx


def evaluate(model, files, max_len, device, temp, max_rows=24):
    model.eval()
    kl_sum = n = top1 = tot = 0.0
    with torch.no_grad():
        for f in files:
            z = np.load(f)
            rows = list(range(min(z["ids"].shape[0], 4)))
            ids, mask, tlog, tidx = batch_from_shard(z, rows, max_len, device)
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits
            lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            logp_s = logits.gather(-1, tidx) - lse
            p_t = torch.softmax(tlog / temp, dim=-1)
            kl = (p_t * (torch.log(p_t + 1e-9) - logp_s)).sum(-1).mean().item()
            top1 += (logits.argmax(-1) == tidx[:, :, 0]).float().sum().item()
            tot += tidx[:, :, 0].numel()
            kl_sum += kl
            n += 1
            if tot >= max_rows * max_len:
                break
    model.train()
    return kl_sum / max(1, n), top1 / max(1, tot)


def main():
    args = parse_args()
    if args.smoke:
        args.max_shards = args.max_shards or 3
        args.max_steps = args.max_steps or 30
        args.eval_every = 0
        args.save_every = 0
        args.max_len = min(args.max_len, 768)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not files:
        raise SystemExit(f"no npz shards in {args.data}")
    eval_files = files[:: args.eval_shard_stride]
    train_files = [f for f in files if f not in set(eval_files)]
    if args.max_shards:
        train_files = train_files[: args.max_shards]

    shapes = [(f, shard_rows(f, args.max_len)) for f in train_files]
    total_rows = sum(n for _, (n, _) in shapes)
    total_tokens = sum(n * L for _, (n, L) in shapes)
    est_steps = math.ceil(total_rows / args.accum) * args.epochs
    print(f"train shards: {len(train_files)} | eval shards: {len(eval_files)}")
    print(f"rows: {total_rows} | tokens(<= {args.max_len}): {total_tokens/1e6:.1f}M | est steps: {est_steps}")

    model, tok, backend = load_student(args)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train/1e6:.2f}M ({100*n_train/sum(p.numel() for p in model.parameters()):.2f}%)")

    import bitsandbytes as bnb

    opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr, betas=(0.9, 0.95))
    warmup = max(10, int(0.03 * est_steps)) if est_steps > 0 else 10

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, est_steps - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    with open(args.log, "w") as fh:
        fh.write("step,epoch,loss,kl,ce,lr,tok_s,vram_mb,elapsed\n")

    device = torch.device("cuda")
    gstep = 0
    t_start = time.time()
    tok_done = 0
    wall_last = t_start
    stop = False

    for epoch in range(args.epochs):
        order = train_files[:]
        random.shuffle(order)
        for f in order:
            if stop:
                break
            z = np.load(f)
            n, L = shard_rows(f, args.max_len)
            rows = list(range(n))
            random.shuffle(rows)
            for i in range(0, n, args.batch):
                sel = rows[i : i + args.batch]
                ids, mask, tlog, tidx = batch_from_shard(z, sel, args.max_len, device)
                loss, kl, ce, _ = compute_loss(model, ids, mask, tlog, tidx, args.alpha, args.temp)
                (loss / args.accum).backward()
                tok_done += ids.numel()
                if (i // args.batch + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    gstep += 1
                    now = time.time()
                    if gstep % 10 == 0 or gstep <= 3:
                        tps = tok_done / max(1e-6, now - wall_last)
                        vram = torch.cuda.max_memory_allocated() / 1e6
                        lr = sched.get_last_lr()[0]
                        print(
                            f"step {gstep} ep{epoch} loss {loss.item():.4f} kl {kl.item():.4f} ce {ce.item():.4f} "
                            f"lr {lr:.2e} tok/s {tps:.0f} vram {vram:.0f}MB",
                            flush=True,
                        )
                        with open(args.log, "a") as fh:
                            fh.write(f"{gstep},{epoch},{loss.item():.4f},{kl.item():.4f},{ce.item():.4f},{lr:.3e},{tps:.1f},{vram:.0f},{now - t_start:.0f}\n")
                        if args.eval_every and gstep % args.eval_every == 0:
                            ek, e1 = evaluate(model, eval_files, args.max_len, device, args.temp)
                            print(f"  eval step {gstep}: KL {ek:.4f} top1 {e1:.3f}", flush=True)
                        if args.save_every and gstep % args.save_every == 0:
                            ck = os.path.join(args.out, f"ckpt-{gstep}")
                            model.save_pretrained(ck)
                            tok.save_pretrained(ck)
                            print("  saved", ck, flush=True)
                    if args.max_steps and gstep >= args.max_steps:
                        stop = True
                        break
    final = os.path.join(args.out, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    with open(os.path.join(args.out, "train_args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"DONE steps={gstep} time={(time.time()-t_start)/60:.1f}min saved={final}")


if __name__ == "__main__":
    main()
