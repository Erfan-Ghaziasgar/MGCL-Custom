import os
import time
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm

from model import SASRec  # SSL not used here
from utils import *  # expects: data_partition, WarpSampler, evaluate, evaluate_valid


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--target_domain', required=True)
parser.add_argument('--source_domain', required=True)
parser.add_argument('--train_dir', default="default")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--beta', default=0.5, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

# New checkpointing-related args
parser.add_argument('--ckpt_dir', default=None, type=str,
                    help='Directory to save checkpoints (default: <target>_<train_dir>/checkpoints)')
parser.add_argument('--save_every', default=50, type=int,
                    help='Save last.ckpt every N epochs regardless of eval cadence')
parser.add_argument('--resume', default=True, type=str2bool,
                    help='Auto-resume from last.ckpt if available')

SEED = 2020
setup_seed(SEED)

args = parser.parse_args()

# Run folder + logging
run_dir = f"{args.target_domain}_{args.train_dir}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, 'args.txt'), 'w') as fargs:
    fargs.write('\n'.join([str(k) + ',' + str(v)
                           for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

# Checkpoint folder
ckpt_dir = args.ckpt_dir or os.path.join(run_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)


def ckpt_path(name: str) -> str:
    return os.path.join(ckpt_dir, name)


def save_checkpoint(epoch, model, optimizer, best_ndcg, total_time, tag='last'):
    payload = {
        'epoch': int(epoch),
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_ndcg': float(best_ndcg),
        'total_time': float(total_time),
        'args': vars(args),
        'rng_state': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    }
    torch.save(payload, ckpt_path(f'{tag}.ckpt'))


def try_load_explicit(model, optimizer):
    """
    Load from --state_dict_path which may be:
      - New .ckpt format (full payload)
      - Old .pth (weights only)
    Returns (epoch_start_idx, best_ndcg, total_time, loaded_bool)
    """
    if not args.state_dict_path:
        return 1, -1.0, 0.0, False

    path = args.state_dict_path
    try:
        state = torch.load(path, map_location=torch.device(args.device))
        # New ckpt format
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
            if 'optimizer_state' in state and optimizer is not None:
                optimizer.load_state_dict(state['optimizer_state'])
            epoch_start_idx = int(state.get('epoch', 0)) + 1
            best_ndcg = float(state.get('best_ndcg', -1.0))
            total_time = float(state.get('total_time', 0.0))
            # Restore RNG (optional)
            rng = state.get('rng_state', {})
            try:
                if rng.get('torch') is not None:
                    torch.set_rng_state(rng['torch'])
                if torch.cuda.is_available() and rng.get('cuda') is not None:
                    torch.cuda.set_rng_state_all(rng['cuda'])
                if rng.get('numpy') is not None:
                    np.random.set_state(rng['numpy'])
                if rng.get('python') is not None:
                    random.setstate(rng['python'])
            except Exception:
                pass
            print(f"Loaded checkpoint (ckpt) from {path} (resume epoch {epoch_start_idx})")
            return epoch_start_idx, best_ndcg, total_time, True

        # Old pure state_dict
        model.load_state_dict(state)
        # Try to infer next epoch from filename pattern epoch=XX
        tail = path[path.find('epoch=') + 6:] if 'epoch=' in path else ''
        try:
            epoch_start_idx = int(tail[:tail.find('.')]) + 1 if '.' in tail else 1
        except Exception:
            epoch_start_idx = 1
        print(f"Loaded model weights (pth) from {path}")
        return epoch_start_idx, -1.0, 0.0, True

    except Exception as e:
        print('Failed loading --state_dict_path:', path, '->', repr(e))
        return 1, -1.0, 0.0, False


def try_autoresume(model, optimizer):
    """
    Auto-resume from last.ckpt if available and --resume true.
    Returns (epoch_start_idx, best_ndcg, total_time, loaded_bool)
    """
    if not args.resume:
        return 1, -1.0, 0.0, False
    last = ckpt_path('last.ckpt')
    if not os.path.isfile(last):
        return 1, -1.0, 0.0, False
    try:
        state = torch.load(last, map_location=torch.device(args.device))
        model.load_state_dict(state['model_state'])
        if optimizer is not None and 'optimizer_state' in state:
            optimizer.load_state_dict(state['optimizer_state'])
        epoch_start_idx = int(state.get('epoch', 0)) + 1
        best_ndcg = float(state.get('best_ndcg', -1.0))
        total_time = float(state.get('total_time', 0.0))
        # Restore RNG (optional)
        rng = state.get('rng_state', {})
        try:
            if rng.get('torch') is not None:
                torch.set_rng_state(rng['torch'])
            if torch.cuda.is_available() and rng.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng['cuda'])
            if rng.get('numpy') is not None:
                np.random.set_state(rng['numpy'])
            if rng.get('python') is not None:
                random.setstate(rng['python'])
        except Exception:
            pass
        print(f"Auto-resumed from {last} (resume epoch {epoch_start_idx}, best_ndcg={best_ndcg:.5f})")
        return epoch_start_idx, best_ndcg, total_time, True
    except Exception as e:
        print('Auto-resume failed:', repr(e))
        return 1, -1.0, 0.0, False


if __name__ == '__main__':
    # === Data ===
    dataset = data_partition(args.target_domain, args.source_domain)
    [user_train1, user_valid1, user_test1, usernum, itemnum1,
     user_neg1, user_train2, user_valid2, user_test2, itemnum2,
     time1, time2] = dataset

    num_batch = max(1, len(user_train1) // args.batch_size)
    cc = 0.0
    for u in user_train1:
        cc += len(user_train1[u])
    print('average sequence length: %.2f' % (cc / len(user_train1)))

    flog = open(os.path.join(run_dir, 'logs_mgcl.txt'), 'a', buffering=1)
    fmetrics = open(os.path.join(run_dir, 'metrics.txt'), 'a', buffering=1)

    # === Sampler & Model ===
    sampler = WarpSampler(user_train1, user_train2, time1, time2, usernum, itemnum1,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=6)
    model = SASRec(user_train1, user_train2, usernum, itemnum1 + itemnum2, args).to(args.device)

    # Param init (safe try)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # === Loss & Optimizer ===
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # === Resume logic ===
    epoch_start_idx, best_ndcg, T = 1, -1.0, 0.0
    loaded = False

    # Explicit path takes precedence
    if args.state_dict_path:
        epoch_start_idx, best_ndcg, T, loaded = try_load_explicit(model, adam_optimizer)
    # Else try auto-resume
    if (not loaded) and args.resume:
        epoch_start_idx, best_ndcg, T, loaded = try_autoresume(model, adam_optimizer)

    # === Inference-only path ===
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        flog.write(f'INFERENCE test: {t_test}\n')
        flog.close()
        fmetrics.close()
        sampler.close()
        raise SystemExit

    # === Training ===
    model.train()
    t0 = time.time()

    try:
        # Precompute GCN embeddings once per epoch (and before resuming progress bar)
        with tqdm(total=args.num_epochs * num_batch, desc="Training", leave=False, ncols=100) as pbar:
            # Fast-forward pbar if resuming mid-run
            if epoch_start_idx > 1:
                pbar.n = (epoch_start_idx - 1) * num_batch
                pbar.refresh()

            for epoch in range(epoch_start_idx, args.num_epochs + 1):
                with torch.no_grad():
                    model.GCN.user_all_embedding, model.GCN.item_all_embedding = model.GCN.forward()

                model.train()
                for _ in range(num_batch):
                    pbar.update(1)
                    pbar.set_postfix({"Epoch": epoch})

                    u, seq, pos, neg, seq2, mask = sampler.next_batch()
                    u, seq, pos, neg, seq2, mask = np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(seq2), np.array(mask)

                    pos_logits, neg_logits, con_loss, con_loss2, con_loss3 = model(u, seq, seq2, pos, neg, mask)
                    pos_labels = torch.ones(pos_logits.shape, device=args.device)
                    neg_labels = torch.zeros(neg_logits.shape, device=args.device)

                    adam_optimizer.zero_grad()
                    indices = np.where(pos != 0)
                    loss1 = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss1 += bce_criterion(neg_logits[indices], neg_labels[indices])

                    loss = loss1 + args.alpha * con_loss + args.beta * con_loss2 + args.gamma * con_loss3
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)
                    loss.backward()
                    adam_optimizer.step()

                # Evaluation cadence (same as your original)
                do_eval = (epoch % 50 == 0 or (epoch % 10 == 0 and epoch >= 400))
                if do_eval:
                    with torch.no_grad():
                        model.GCN.user_all_embedding, model.GCN.item_all_embedding = model.GCN.forward()
                    model.eval()
                    t1 = time.time() - t0
                    T += t1

                    t_test = evaluate(model, dataset, args)
                    t_valid = evaluate_valid(model, dataset, args)

                    msg = ('epoch:%d, time: %f(s), valid (NDCG@10: %.5f, HR@10: %.5f), '
                           'test (NDCG@10: %.5f, HR@10: %.5f)') % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
                    print(msg)

                    flog.write(str(t_valid) + ' ' + str(t_test) + '\n'); flog.flush()
                    fmetrics.write(msg + '\n'); fmetrics.flush()
                    t0 = time.time()

                    # Save best by validation NDCG
                    if t_valid[0] > best_ndcg:
                        best_ndcg = t_valid[0]
                        save_checkpoint(epoch, model, adam_optimizer, best_ndcg, T, tag='best')

                    # Always refresh last.ckpt after eval
                    save_checkpoint(epoch, model, adam_optimizer, best_ndcg, T, tag='last')

                # Also snapshot every N epochs regardless of eval cadence
                if args.save_every and (epoch % args.save_every == 0):
                    save_checkpoint(epoch, model, adam_optimizer, best_ndcg, T, tag='last')

            # Final saves
            final_name = 'MGCL.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'.format(
                args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(run_dir, final_name))
            save_checkpoint(args.num_epochs, model, adam_optimizer, best_ndcg, T, tag='last')

    except KeyboardInterrupt:
        print('\nInterrupted — saving last checkpoint before exit.')
        try:
            # Best effort: epoch may be undefined if interrupt very early
            current_epoch = max(1, locals().get('epoch', 1))
            save_checkpoint(current_epoch, model, adam_optimizer, best_ndcg, T, tag='last')
        except Exception as e:
            print('Failed to save on interrupt:', repr(e))
        raise
    finally:
        flog.close()
        fmetrics.close()
        sampler.close()
        print("Done")
