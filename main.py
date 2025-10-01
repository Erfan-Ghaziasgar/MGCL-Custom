#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import argparse
import random
import numpy as np
from tqdm.auto import tqdm

from model import SASRec
from utils import (
    WarpSampler,
    data_partition,
    evaluate,
    evaluate_valid,
)

# ----------------------------
# CLI & setup utilities
# ----------------------------
def str2bool(s: str) -> bool:
    s = str(s).lower()
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def ckpt_path(ckpt_dir: str, name: str) -> str:
    return os.path.join(ckpt_dir, name)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MGCL with Semantic Integration (refactor)")
    # Domains & I/O
    p.add_argument("--target_domain", required=True)
    p.add_argument("--source_domain", required=True)
    p.add_argument("--train_dir", default="default")
    p.add_argument("--ckpt_dir", default=None, type=str)
    p.add_argument("--save_every", default=50, type=int)
    p.add_argument("--resume", default=True, type=str2bool)
    p.add_argument("--state_dict_path", default=None, type=str)
    p.add_argument("--inference_only", default=False, type=str2bool)

    # Model
    p.add_argument("--maxlen", default=100, type=int)
    p.add_argument("--hidden_units", default=50, type=int)
    p.add_argument("--num_blocks", default=2, type=int)
    p.add_argument("--num_heads", default=1, type=int)
    p.add_argument("--dropout_rate", default=0.5, type=float)

    # Train
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--l2_emb", default=1e-3, type=float)
    p.add_argument("--num_epochs", default=500, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", default=6, type=int)

    # Loss weights
    p.add_argument("--alpha", default=0.5, type=float)
    p.add_argument("--beta", default=0.5, type=float)
    p.add_argument("--gamma", default=0.5, type=float)
    p.add_argument("--delta", default=0.2, type=float)

    # Semantic loss temp (for SSL_semantic)
    p.add_argument("--temp", default=0.1, type=float)
    return p


def save_checkpoint(args, ckpt_dir, epoch, model, optimizer, best_ndcg, best_hr, total_time, tag="last"):
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_ndcg": float(best_ndcg),
        "best_hr": float(best_hr),
        "total_time": float(total_time),
        "args": vars(args),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    path = ckpt_path(ckpt_dir, f"{tag}.ckpt")
    torch.save(payload, path)


def try_load_checkpoint(args, model, optimizer, path):
    """
    Return: (epoch_start_idx, best_ndcg, best_hr, total_time, loaded)
    """
    if not path or not os.path.isfile(path):
        return 1, -1.0, -1.0, 0.0, False
    try:
        # PyTorch 2.6 defaults to weights_only=True; we need full state (safe if you trust the file).
        state = torch.load(path, map_location=torch.device(args.device), weights_only=False)

        model.load_state_dict(state["model_state"])
        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])

        epoch = int(state.get("epoch", 0)) + 1
        best_ndcg = float(state.get("best_ndcg", -1.0))
        best_hr = float(state.get("best_hr", -1.0))
        total_time = float(state.get("total_time", 0.0))

        # Restore RNG (optional but nice for reproducibility)
        rng = state.get("rng_state", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])

        return epoch, best_ndcg, best_hr, total_time, True
    except Exception:
        # Fall back to fresh start if anything goes wrong
        return 1, -1.0, -1.0, 0.0, False


def train_epoch(model, sampler, optimizer, bce_criterion, num_batch, args, pbar):
    """
    One epoch over num_batch training batches.
    Uses a single global tqdm 'pbar' — no extra bars, minimal updates.
    """
    model.train()

    epoch_loss = epoch_bpr = epoch_con = epoch_sem = 0.0

    for b in range(num_batch):
        u, seq, pos, neg, seq2, mask = sampler.next_batch()
        u = np.asarray(u); seq = np.asarray(seq); pos = np.asarray(pos)
        neg = np.asarray(neg); seq2 = np.asarray(seq2); mask = np.asarray(mask)

        pos_logits, neg_logits, con_loss, con_loss2, con_loss3, semantic_con_loss = model(
            u, seq, seq2, pos, neg, mask
        )

        # BPR-style BCE on observed/non-observed pairs (ignore pads)
        indices = np.where(pos != 0)
        loss_bpr = bce_criterion(pos_logits[indices], torch.ones_like(pos_logits[indices])) + \
                   bce_criterion(neg_logits[indices], torch.zeros_like(neg_logits[indices]))

        contrastive = args.alpha * con_loss + args.beta * con_loss2 + args.gamma * con_loss3
        loss = loss_bpr + contrastive + args.delta * semantic_con_loss

        if args.l2_emb > 0:
            loss = loss + args.l2_emb * model.item_emb.weight.norm(2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # running means
        epoch_loss += loss.item()
        epoch_bpr += loss_bpr.item()
        epoch_con += contrastive.item()
        epoch_sem += (args.delta * semantic_con_loss).item()

        # single global bar update (throttled postfix)
        pbar.update(1)
        if (b + 1) % 20 == 0 or (b + 1) == num_batch:
            avg_batches = b + 1
            pbar.set_postfix_str(
                f"loss={epoch_loss/avg_batches:.2f} bpr={epoch_bpr/avg_batches:.3f} con={epoch_con/avg_batches:.2f} sem={epoch_sem/avg_batches:.2f}",
                refresh=False
            )

    return {
        "total": epoch_loss / num_batch,
        "bpr": epoch_bpr / num_batch,
        "contrastive": epoch_con / num_batch,
        "semantic": epoch_sem / num_batch,
    }


def main():
    setup_seed(2020)
    parser = get_parser()
    args = parser.parse_args()

    # Paths
    run_dir = f"{args.target_domain}_{args.train_dir}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = args.ckpt_dir or os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Persist args (quietly)
    with open(os.path.join(run_dir, "args.txt"), "w") as f:
        f.write("\n".join([f"{k},{v}" for k, v in sorted(vars(args).items())]))

    # ----------------------------
    # Load data
    # ----------------------------
    print("Loading data...", flush=True)
    dataset = data_partition(args.target_domain, args.source_domain)
    (
        user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1,
        user_train2, user_valid2, user_test2, itemnum2,
        time1, time2, item_text_embeddings, item_id_to_idx, reindexed_to_original
    ) = dataset

    num_batch = max(1, len(user_train1) // args.batch_size)

    # ----------------------------
    # Model & Sampler
    # ----------------------------
    sampler = WarpSampler(
        user_train1, user_train2, time1, time2, usernum, itemnum1,
        batch_size=args.batch_size, maxlen=args.maxlen, n_workers=args.num_workers
    )

    model = SASRec(
        user_train1, user_train2, usernum, itemnum1 + itemnum2,
        item_text_embeddings, item_id_to_idx, reindexed_to_original, args
    ).to(args.device)

    # Xavier for trainable matrices (skip text table)
    for name, p in model.named_parameters():
        if p.dim() > 1 and p.requires_grad:
            torch.nn.init.xavier_normal_(p.data)

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # ----------------------------
    # Resume if requested
    # ----------------------------
    epoch_start_idx, best_ndcg, best_hr, T = 1, -1.0, -1.0, 0.0
    if args.resume:
        load_path = args.state_dict_path if (args.state_dict_path and os.path.isfile(args.state_dict_path)) \
                    else ckpt_path(ckpt_dir, "last.ckpt")
        epoch_start_idx, best_ndcg, best_hr, T, loaded = try_load_checkpoint(args, model, optimizer, load_path)
        if loaded:
            print(f"Resumed from '{load_path}' @ epoch {epoch_start_idx} "
                  f"(best: NDCG@10={best_ndcg:.5f}, HR@10={best_hr:.5f})", flush=True)
        else:
            print("No usable checkpoint found — starting fresh.", flush=True)
    else:
        print("Resume disabled — starting fresh.", flush=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready. Trainable params: {trainable_params}", flush=True)

    # ----------------------------
    # Inference-only mode
    # ----------------------------
    if args.inference_only:
        print("Inference only...", flush=True)
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f"Test: NDCG@10={t_test[0]:.5f}, HR@10={t_test[1]:.5f}", flush=True)
        sampler.close()
        return

    # ----------------------------
    # Training loop (single global tqdm)
    # ----------------------------
    print("Training started.", flush=True)
    try:
        with tqdm(
            total=args.num_epochs * num_batch,
            desc="Training",
            initial=(epoch_start_idx - 1) * num_batch,
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ) as pbar:

            for epoch in range(epoch_start_idx, args.num_epochs + 1):
                t0 = time.time()
                _ = train_epoch(model, sampler, optimizer, bce_criterion, num_batch, args, pbar)
                epoch_time = time.time() - t0
                T += epoch_time

                # evaluate sparsely
                should_eval = (epoch % args.save_every == 0) or (epoch == args.num_epochs)
                if should_eval:
                    model.eval()
                    t_valid = evaluate_valid(model, dataset, args)
                    t_test = evaluate(model, dataset, args)
                    print(
                        f"[Epoch {epoch:3d}] "
                        f"time={epoch_time:.1f}s | "
                        f"valid NDCG@10={t_valid[0]:.5f}, HR@10={t_valid[1]:.5f} | "
                        f"test NDCG@10={t_test[0]:.5f}, HR@10={t_test[1]:.5f}",
                        flush=True
                    )

                    improved = False
                    if t_valid[0] > best_ndcg:
                        best_ndcg, best_hr = t_valid[0], t_valid[1]
                        save_checkpoint(args, ckpt_dir, epoch, model, optimizer, best_ndcg, best_hr, T, tag="best")
                        print(f"  ↳ new best: NDCG@10={best_ndcg:.5f}, HR@10={best_hr:.5f}", flush=True)
                        improved = True

                    save_checkpoint(args, ckpt_dir, epoch, model, optimizer, best_ndcg, best_hr, T, tag="last")
                    if not improved:
                        # keep output tidy; just acknowledge save
                        print("  ↳ checkpoint saved (last)", flush=True)
                    model.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted — saving...", flush=True)

    finally:
        # Always save a last snapshot on exit
        save_checkpoint(args, ckpt_dir, epoch, model, optimizer, best_ndcg, best_hr, T, tag="last")
        sampler.close()

        print(
            f"Done. Best valid NDCG@10={best_ndcg:.5f}, HR@10={best_hr:.5f} | "
            f"Total train time={T/3600:.2f}h | ckpts: {ckpt_dir}",
            flush=True
        )


if __name__ == "__main__":
    main()
