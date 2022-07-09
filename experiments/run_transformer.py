# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import copy
import pandas as pd
from collections import defaultdict
import torch


import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric import datasets
from transformer.models import DiffGraphTransformer, GraphTransformer
from transformer.data import GraphDataset
from transformer.position_encoding import LapEncoding, FullEncoding, POSENCODINGS
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_args():
    parser = argparse.ArgumentParser(
        description='Transformer baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="ZINC",
                        help='name of dataset')
    parser.add_argument('--nb-heads', type=int, default=8)
    parser.add_argument('--nb-layers', type=int, default=10)
    parser.add_argument('--dim-hidden', type=int, default=64)
    parser.add_argument('--pos-enc', choices=[None,
                        'diffusion', 'pstep', 'adj'], default=None)
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE')
    parser.add_argument('--lap-dim', type=int, default=8, help='dimension for laplacian PE')
    parser.add_argument('--p', type=int, default=1, help='p step random walk kernel')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='bandwidth for the diffusion kernel')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym',
                        help='normalization for Laplacian')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--zero-diag', action='store_true', help='zero diagonal for PE matrix')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/transformer'
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.zero_diag:
            outdir = outdir + '/zero_diag'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        lapdir = 'NoPE' if not args.lappe else 'Lap_{}'.format(args.lap_dim) 
        outdir = outdir + '/{}'.format(lapdir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.lr, args.nb_layers, args.nb_heads, args.dim_hidden, bn,
            args.pos_enc, args.normalization, args.p, args.beta
        )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0

    tic = timer()
    for i, (data, mask, pe, lap_pe, degree, labels) in enumerate(loader):
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.lappe:
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(lap_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            lap_pe = lap_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()
            mask = mask.cuda()
            if pe is not None:
                pe = pe.cuda()
            if lap_pe is not None:
                lap_pe = lap_pe.cuda()
            if degree is not None:
                degree = degree.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        import time
        t1 = time.time()
        output = model(data, mask, pe, lap_pe, degree)
        t2 = time.time()
        loss = criterion(output, labels)
        t3 = time.time()
        loss.backward()
        t4 = time.time()
        optimizer.step()
        t5 = time.time()

        
        print('== ',t2-t1,t3-t2,t4-t3,t5-t4)
        running_loss += loss.item() * len(data)

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()
    with torch.no_grad():
        for data, mask, pe, lap_pe, degree, labels in loader:
            if args.lappe:
                # sign flip as in Bresson et al. for laplacian PE
                sign_flip = torch.rand(lap_pe.shape[-1])
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                lap_pe = lap_pe * sign_flip.unsqueeze(0)

            if use_cuda:
                data = data.cuda()
                mask = mask.cuda()
                if pe is not None:
                    pe = pe.cuda()
                if lap_pe is not None:
                    lap_pe = lap_pe.cuda()
                if degree is not None:
                    degree = degree.cuda()
                labels = labels.cuda()

            output = model(data, mask, pe, lap_pe, degree)
            loss = criterion(output, labels)
            mse_loss += F.mse_loss(output, labels).item() * len(data)
            mae_loss += F.l1_loss(output, labels).item() * len(data)

            running_loss += loss.item() * len(data)
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_mse = mse_loss / n_sample
    print('Val loss: {:.4f} MSE loss: {:.4f} MAE loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_mse, epoch_mae, toc - tic))
    return epoch_mae, epoch_mse

def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../dataset/ZINC'
    # number of node attributes for ZINC dataset
    n_tags = 28

    train_dset = GraphDataset(datasets.ZINC(data_path, subset=True, split='train'), n_tags, degree=True)
    input_size = train_dset.input_size()
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dset.collate_fn())
    print(len(train_dset))
    print(train_dset[0])

    val_dset = GraphDataset(datasets.ZINC(data_path, subset=True, split='val'), n_tags, degree=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dset.collate_fn())

    pos_encoder = None
    if args.pos_enc is not None:
        pos_encoding_method = POSENCODINGS.get(args.pos_enc, None)
        pos_encoding_params_str = ""
        if args.pos_enc == 'diffusion':
            pos_encoding_params = {
                'beta': args.beta
            }
            pos_encoding_params_str = args.beta
        elif args.pos_enc == 'pstep':
            pos_encoding_params = {
                'beta': args.beta,
                'p': args.p
            }
            pos_encoding_params_str = "{}_{}".format(args.p, args.beta)
        else:
            pos_encoding_params = {}

        if pos_encoding_method is not None:
            pos_cache_path = '../cache/pe/zinc_{}_{}_{}.pkl'.format(args.pos_enc, args.normalization, pos_encoding_params_str)
            pos_encoder = pos_encoding_method(pos_cache_path, normalization=args.normalization, zero_diag=args.zero_diag, **pos_encoding_params)

        print("Position encoding...")
        pos_encoder.apply_to(train_dset, split='train')
        pos_encoder.apply_to(val_dset, split='val')
    else:
        if args.zero_diag:
            pos_encoder = FullEncoding(None, args.zero_diag)
            pos_encoder.apply_to(train_dset, split='train')
            pos_encoder.apply_to(val_dset, split='val')

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
        lap_pos_encoder.apply_to(train_dset)
        lap_pos_encoder.apply_to(val_dset)

    if args.pos_enc is not None:
        model = DiffGraphTransformer(in_size=input_size,
                                     nb_class=1,
                                     d_model=args.dim_hidden,
                                     dim_feedforward=2*args.dim_hidden,
                                     dropout=args.dropout,
                                     nb_heads=args.nb_heads,
                                     nb_layers=args.nb_layers,
                                     batch_norm=args.batch_norm,
                                     lap_pos_enc=args.lappe,
                                     lap_pos_enc_dim=args.lap_dim)
    else:
        model = GraphTransformer(in_size=input_size,
                                 nb_class=1,
                                 d_model=args.dim_hidden,
                                 dim_feedforward=2*args.dim_hidden,
                                 dropout=args.dropout,
                                 nb_heads=args.nb_heads,
                                 nb_layers=args.nb_layers,
                                 lap_pos_enc=args.lappe,
                                 lap_pos_enc_dim=args.lap_dim)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=15,
                                                     min_lr=1e-05,
                                                     verbose=False)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    test_dset = GraphDataset(datasets.ZINC(data_path, subset=True, split='test'), n_tags, degree=True)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dset.collate_fn())
    if pos_encoder is not None:
        pos_encoder.apply_to(test_dset, split='test')

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder.apply_to(test_dset)

    print("Training...")
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
        val_loss,_ = eval_epoch(model, val_loader, criterion, args.use_cuda)
        test_loss,_ = eval_epoch(model, test_loader, criterion, args.use_cuda)

        if args.warmup is None:
            lr_scheduler.step(val_loss)

        logs['train_mae'].append(train_loss)
        logs['val_mae'].append(val_loss)
        logs['test_mae'].append(test_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_loss, test_mse_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

    print("test MAE loss {:.4f}".format(test_loss))

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_mae': test_loss,
            'test_mse': test_mse_loss,
            'val_mae': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model.pkl')


if __name__ == "__main__":
    main()
