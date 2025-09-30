import os
import time
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm

from model import SASRec
from utils import *


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
parser.add_argument('--delta', default=0.1, type=float, help='Weight for the semantic contrastive loss.')
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--ckpt_dir', default=None, type=str)
parser.add_argument('--save_every', default=50, type=int)
parser.add_argument('--resume', default=True, type=str2bool)
parser.add_argument('--num_workers', default=6, type=int)


def ckpt_path(ckpt_dir, name: str) -> str:
    return os.path.join(ckpt_dir, name)


def save_checkpoint(ckpt_dir, epoch, model, optimizer, best_ndcg, total_time, tag='last'):
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
    torch.save(payload, ckpt_path(ckpt_dir, f'{tag}.ckpt'))
    print(f"Saved {tag} checkpoint at epoch {epoch}.")


def try_load_checkpoint(model, optimizer, ckpt_dir, path):
    if not path or not os.path.isfile(path):
        print("No checkpoint found to resume from.")
        return 1, -1.0, 0.0, False
    try:
        state = torch.load(path, map_location=torch.device(args.device))
        model.load_state_dict(state['model_state'])
        if 'optimizer_state' in state and optimizer is not None:
            optimizer.load_state_dict(state['optimizer_state'])
        epoch = int(state.get('epoch', 0)) + 1
        best_ndcg = float(state.get('best_ndcg', -1.0))
        total_time = float(state.get('total_time', 0.0))
        print(f"Loaded checkpoint from {path} (resuming at epoch {epoch})")
        return epoch, best_ndcg, total_time, True
    except Exception as e:
        print(f'Failed to load checkpoint from {path}: {repr(e)}')
        return 1, -1.0, 0.0, False


if __name__ == '__main__':
    SEED = 2020
    setup_seed(SEED)
    args = parser.parse_args()

    run_dir = f"{args.target_domain}_{args.train_dir}"
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([f"{k},{v}" for k, v in sorted(vars(args).items())]))

    ckpt_dir = args.ckpt_dir or os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading and partitioning data...")
    dataset = data_partition(args.target_domain, args.source_domain)
    
    [user_train1, user_valid1, user_test1, usernum, itemnum1,
     user_neg1, user_train2, user_valid2, user_test2, itemnum2,
     time1, time2, item_text_embeddings, item_id_to_idx] = dataset
    
    num_batch = max(1, len(user_train1) // args.batch_size)
    print(f'Average sequence length: {sum(len(v) for v in user_train1.values()) / len(user_train1):.2f}')

    flog = open(os.path.join(run_dir, 'logs_mgcl.txt'), 'a', buffering=1)
    fmetrics = open(os.path.join(run_dir, 'metrics.txt'), 'a', buffering=1)

    sampler = WarpSampler(user_train1, user_train2, time1, time2, usernum, itemnum1,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=args.num_workers)
    
    model = SASRec(user_train1, user_train2, usernum, itemnum1 + itemnum2, 
                   item_text_embeddings, item_id_to_idx, args).to(args.device)

    for name, param in model.named_parameters():
        if 'text_emb' not in name and param.dim() > 1:
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    epoch_start_idx, best_ndcg, T = 1, -1.0, 0.0
    
    if args.resume:
        # Prioritize loading a specific manual path, otherwise fall back to the last auto-saved checkpoint.
        load_path = args.state_dict_path if (args.state_dict_path and os.path.isfile(args.state_dict_path)) else ckpt_path(ckpt_dir, 'last.ckpt')
        epoch_start_idx, best_ndcg, T, _ = try_load_checkpoint(model, adam_optimizer, ckpt_dir, load_path)

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Inference results -- Test (NDCG@10: {t_test[0]:.5f}, HR@10: {t_test[1]:.5f})')
        flog.write(f"INFERENCE,{t_test[0]},{t_test[1]}\n")
        sampler.close(); flog.close(); fmetrics.close()
        exit()

    try:
        with tqdm(total=args.num_epochs * num_batch, desc="Training", initial=(epoch_start_idx - 1) * num_batch) as pbar:
            for epoch in range(epoch_start_idx, args.num_epochs + 1):
                model.train()
                
                with torch.no_grad():
                    model.GCN.user_all_embedding, model.GCN.item_all_embedding = model.GCN.forward()

                for _ in range(num_batch):
                    u, seq, pos, neg, seq2, mask = sampler.next_batch()
                    u, seq, pos, neg, seq2, mask = [np.array(x) for x in [u, seq, pos, neg, seq2, mask]]
                    
                    pos_logits, neg_logits, con_loss, con_loss2, con_loss3, semantic_con_loss = model(u, seq, seq2, pos, neg, mask)
                    
                    indices = np.where(pos != 0)
                    loss_bpr = bce_criterion(pos_logits[indices], torch.ones_like(pos_logits[indices])) + \
                               bce_criterion(neg_logits[indices], torch.zeros_like(neg_logits[indices]))
                    
                    loss = loss_bpr + \
                           args.alpha * con_loss + \
                           args.beta * con_loss2 + \
                           args.gamma * con_loss3 + \
                           args.delta * semantic_con_loss
                    
                    if args.l2_emb > 0:
                        loss += args.l2_emb * model.item_emb.weight.norm(2)

                    adam_optimizer.zero_grad()
                    loss.backward()
                    adam_optimizer.step()
                    pbar.update(1)
                    pbar.set_postfix({"Epoch": epoch, "Loss": f"{loss.item():.4f}"})
                
                if (epoch % args.save_every == 0) or (epoch >= 400 and epoch % 10 == 0):
                    model.eval()
                    print("\nEvaluating...")
                    t_valid = evaluate_valid(model, dataset, args)
                    t_test = evaluate(model, dataset, args)
                    
                    msg = (f'epoch:{epoch}, valid (NDCG@10: {t_valid[0]:.5f}, HR@10: {t_valid[1]:.5f}), '
                           f'test (NDCG@10: {t_test[0]:.5f}, HR@10: {t_test[1]:.5f})')
                    print(msg)
                    flog.write(f"({t_valid[0]}, {t_valid[1]}) ({t_test[0]}, {t_test[1]})\n")
                    fmetrics.write(msg + '\n')
                    
                    if t_valid[0] > best_ndcg:
                        best_ndcg = t_valid[0]
                        save_checkpoint(ckpt_dir, epoch, model, adam_optimizer, best_ndcg, T, tag='best')
                    
                    save_checkpoint(ckpt_dir, epoch, model, adam_optimizer, best_ndcg, T, tag='last')

    except KeyboardInterrupt:
        print('\nTraining interrupted. Saving final state...')
    finally:
        # Ensure a final checkpoint is saved upon completion or interruption
        save_checkpoint(ckpt_dir, epoch, model, adam_optimizer, best_ndcg, T, tag='last')
        flog.close()
        fmetrics.close()
        sampler.close()
        print("Training complete.")

