#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
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


@dataclass
class TrainingState:
    """Minimal mutable container for tracking progress."""

    epoch: int = 1
    best_ndcg: float = -1.0
    best_hr: float = -1.0
    elapsed: float = 0.0

    def register_epoch(self, epoch: int, duration: float) -> None:
        self.epoch = epoch
        self.elapsed += float(duration)

    def maybe_update_best(self, ndcg: float, hr: float) -> bool:
        improved = ndcg > self.best_ndcg
        if improved:
            self.best_ndcg = float(ndcg)
            self.best_hr = float(hr)
        return improved


def save_checkpoint(
    args,
    ckpt_dir: Path,
    state: TrainingState,
    model,
    optimizer,
    tag: str = "last",
) -> None:
    payload = {
        "epoch": int(state.epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_ndcg": float(state.best_ndcg),
        "best_hr": float(state.best_hr),
        "total_time": float(state.elapsed),
        "args": vars(args),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    path = ckpt_dir / f"{tag}.ckpt"
    torch.save(payload, str(path))


def try_load_checkpoint(
    args,
    model,
    optimizer,
    path: Optional[Path],
) -> Tuple[int, TrainingState, bool]:
    """Attempt to load a checkpoint, returning the next epoch and state."""

    if not path or not path.is_file():
        return 1, TrainingState(), False
    try:
        # PyTorch 2.6 defaults to weights_only=True; we need full state (safe if you trust the file).
        payload = torch.load(str(path), map_location=torch.device(args.device), weights_only=False)

        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])

        state = TrainingState(
            epoch=int(payload.get("epoch", 0)) + 1,
            best_ndcg=float(payload.get("best_ndcg", -1.0)),
            best_hr=float(payload.get("best_hr", -1.0)),
            elapsed=float(payload.get("total_time", 0.0)),
        )

        # Restore RNG (optional but nice for reproducibility)
        rng = payload.get("rng_state", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])

        return state.epoch, state, True
    except Exception:
        # Fall back to fresh start if anything goes wrong
        return 1, TrainingState(), False


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


def prepare_run_directories(args) -> Tuple[Path, Path]:
    run_dir = Path(f"{args.target_domain}_{args.train_dir}")
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir


def persist_run_configuration(args, run_dir: Path) -> None:
    args_txt = "\n".join(f"{k},{v}" for k, v in sorted(vars(args).items()))
    (run_dir / "args.txt").write_text(args_txt)


def load_dataset(args):
    print("Loading data...", flush=True)
    return data_partition(args.target_domain, args.source_domain)


def build_sampler(dataset, args) -> WarpSampler:
    (
        user_train1,
        _user_valid1,
        _user_test1,
        usernum,
        itemnum1,
        _user_neg1,
        user_train2,
        _user_valid2,
        _user_test2,
        _itemnum2,
        time1,
        time2,
        *_,
    ) = dataset

    return WarpSampler(
        user_train1,
        user_train2,
        time1,
        time2,
        usernum,
        itemnum1,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=args.num_workers,
    )


def build_model(dataset, args) -> SASRec:
    (
        user_train1,
        _user_valid1,
        _user_test1,
        usernum,
        itemnum1,
        _user_neg1,
        user_train2,
        _user_valid2,
        _user_test2,
        itemnum2,
        _time1,
        _time2,
        item_text_embeddings,
        item_id_to_idx,
        reindexed_to_original,
    ) = dataset

    model = SASRec(
        user_train1,
        user_train2,
        usernum,
        itemnum1 + itemnum2,
        item_text_embeddings,
        item_id_to_idx,
        reindexed_to_original,
        args,
    ).to(args.device)

    for _, parameter in model.named_parameters():
        if parameter.dim() > 1 and parameter.requires_grad:
            torch.nn.init.xavier_normal_(parameter.data)

    return model


def resume_if_requested(args, model, optimizer, ckpt_dir: Path) -> Tuple[int, TrainingState]:
    if not args.resume:
        print("Resume disabled — starting fresh.", flush=True)
        return 1, TrainingState()

    load_path = None
    if args.state_dict_path:
        candidate = Path(args.state_dict_path)
        if candidate.is_file():
            load_path = candidate
    if load_path is None:
        load_path = ckpt_dir / "last.ckpt"

    start_epoch, state, loaded = try_load_checkpoint(args, model, optimizer, load_path)
    if loaded:
        print(
            f"Resumed from '{load_path}' @ epoch {start_epoch} "
            f"(best: NDCG@10={state.best_ndcg:.5f}, HR@10={state.best_hr:.5f})",
            flush=True,
        )
    else:
        print("No usable checkpoint found — starting fresh.", flush=True)
    return start_epoch, state


def run_inference_only(model, dataset, sampler, args) -> None:
    print("Inference only...", flush=True)
    model.eval()
    t_test = evaluate(model, dataset, args)
    print(f"Test: NDCG@10={t_test[0]:.5f}, HR@10={t_test[1]:.5f}", flush=True)
    sampler.close()


def compute_num_batches(user_train1, batch_size: int) -> int:
    user_count = max(1, len(user_train1))
    return max(1, user_count // max(1, batch_size))


def run_training(
    args,
    dataset,
    model,
    sampler,
    optimizer,
    criterion,
    num_batch: int,
    start_epoch: int,
    state: TrainingState,
    ckpt_dir: Path,
) -> None:
    print("Training started.", flush=True)
    last_epoch = start_epoch - 1

    try:
        with tqdm(
            total=args.num_epochs * num_batch,
            desc="Training",
            initial=(start_epoch - 1) * num_batch,
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for epoch in range(start_epoch, args.num_epochs + 1):
                last_epoch = epoch
                t0 = time.time()
                train_epoch(model, sampler, optimizer, criterion, num_batch, args, pbar)
                epoch_time = time.time() - t0
                state.register_epoch(epoch, epoch_time)

                should_eval = (epoch % args.save_every == 0) or (epoch == args.num_epochs)
                if not should_eval:
                    continue

                model.eval()
                t_valid = evaluate_valid(model, dataset, args)
                t_test = evaluate(model, dataset, args)
                print(
                    f"[Epoch {epoch:3d}] "
                    f"time={epoch_time:.1f}s | "
                    f"valid NDCG@10={t_valid[0]:.5f}, HR@10={t_valid[1]:.5f} | "
                    f"test NDCG@10={t_test[0]:.5f}, HR@10={t_test[1]:.5f}",
                    flush=True,
                )

                if state.maybe_update_best(t_valid[0], t_valid[1]):
                    save_checkpoint(args, ckpt_dir, state, model, optimizer, tag="best")
                    print(
                        f"  ↳ new best: NDCG@10={state.best_ndcg:.5f}, HR@10={state.best_hr:.5f}",
                        flush=True,
                    )

                save_checkpoint(args, ckpt_dir, state, model, optimizer, tag="last")
                print("  ↳ checkpoint saved (last)", flush=True)
                model.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted — saving...", flush=True)

    finally:
        state.epoch = last_epoch
        save_checkpoint(args, ckpt_dir, state, model, optimizer, tag="last")
        sampler.close()

        print(
            f"Done. Best valid NDCG@10={state.best_ndcg:.5f}, HR@10={state.best_hr:.5f} | "
            f"Total train time={state.elapsed/3600:.2f}h | ckpts: {ckpt_dir}",
            flush=True,
        )


def main():
    setup_seed(2020)
    parser = get_parser()
    args = parser.parse_args()

    run_dir, ckpt_dir = prepare_run_directories(args)
    persist_run_configuration(args, run_dir)

    dataset = load_dataset(args)
    user_train1 = dataset[0]

    num_batch = compute_num_batches(user_train1, args.batch_size)

    sampler = build_sampler(dataset, args)
    model = build_model(dataset, args)

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    start_epoch, state = resume_if_requested(args, model, optimizer, ckpt_dir)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready. Trainable params: {trainable_params}", flush=True)

    if args.inference_only:
        run_inference_only(model, dataset, sampler, args)
        return

    run_training(
        args=args,
        dataset=dataset,
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        criterion=bce_criterion,
        num_batch=num_batch,
        start_epoch=start_epoch,
        state=state,
        ckpt_dir=ckpt_dir,
    )


if __name__ == "__main__":
    main()
