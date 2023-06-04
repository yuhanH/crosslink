import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset import TFBindingDataset
from model import TFBindingModel
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from datetime import datetime
import argparse
import torch.distributed as dist

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--split', type=str, default='tf_in_domain', choices=['random', 'chr', 'tf_cluster', 'tf_and_chr', 'tf_in_domain', 'tf_in_domain_and_chr'])
    parser.add_argument('--run_dir', type=str, default='/home/ubuntu/codebase/tf_binding/runs/')
    parser.add_argument('--epoches', type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # Initialize distributed training
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Run ID
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Load data and model
    train_dataset = TFBindingDataset(mode='train', split=args.split)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=30)
    val_dataset = TFBindingDataset(mode='val', split=args.split)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=10)
    model = TFBindingModel().cuda()
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loop stats and metrics
    train_loss_list = []
    val_loss_list = []
    pearsonr_list = []
    spearmanr_list = []
    lowest_val_loss = float('inf')

    # Train model
    for epoch in range(args.epoches):
        # Train loop
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            seq, tf_embedding, label = batch
            optimizer.zero_grad()
            pred = model(seq.float().cuda(), tf_embedding.float().cuda()).view(-1)
            loss = criterion(pred, label.float().cuda())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss_list.append(train_loss / len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0
        pred_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                seq, tf_embedding, label = batch
                pred = model(seq.float().cuda(), tf_embedding.float().cuda()).view(-1)
                loss = criterion(pred, label.float().cuda())

                val_loss += loss.item()
                pred_list.append(pred.cpu().numpy())
                label_list.append(label.numpy())

        val_loss_list.append(val_loss / len(val_loader))
        pred_list = np.concatenate(pred_list)
        label_list = np.concatenate(label_list)
        pearsonr_val = pearsonr(pred_list, label_list)[0]
        spearmanr_val = spearmanr(pred_list, label_list)[0]
        pearsonr_list.append(pearsonr_val)
        spearmanr_list.append(spearmanr_val)
        print(f'Epoch {epoch}, train loss {train_loss_list[-1]}, val loss {val_loss_list[-1]}, pearsonr {pearsonr_val}, spearmanr {spearmanr_val}')

        # Save model
        save_path = f'{args.run_dir}{run_id}'
        if val_loss_list[-1] < lowest_val_loss:
            lowest_val_loss = val_loss_list[-1]
            os.makedirs(f'{save_path}/models', exist_ok=True)
            torch.save(model.module.state_dict(), f'{save_path}/models/model_{epoch}.pt')

        # Save stats to text file
        with open(f'{save_path}/stats.txt', 'w') as f:
            f.write(f'Train loss: {train_loss_list}\n')
            f.write(f'Val loss: {val_loss_list}\n')
            f.write(f'Pearsonr: {pearsonr_list}\n')
            f.write(f'Spearmanr: {spearmanr_list}\n')

if __name__ == '__main__':
    main()