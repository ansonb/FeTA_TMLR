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
from transformer.models import DiffGraphTransformer, GraphTransformer, DiffGraphTransformerGCN, DiffGraphTransformerGenGCNSBM
from transformer.data import GraphDataset, GraphDataset_sbm
from transformer.position_encoding import LapEncoding, POSENCODINGS
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim

from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer import utils

def load_args():
    parser = argparse.ArgumentParser(
        description='Transformer baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="PATTERN",
                        help='name of dataset')
    parser.add_argument('--nb-heads', type=int, default=4)
    parser.add_argument('--nb-layers', type=int, default=3)
    parser.add_argument('--dim-hidden', type=int, default=64)
    parser.add_argument('--pos-enc', choices=[None,
                        'diffusion', 'pstep', 'adj'], default=None)
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE')
    parser.add_argument('--lap-dim', type=int, default=2, help='dimension for laplacian PE')
    parser.add_argument('--p', type=int, default=1, help='p step random walk kernel')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='bandwidth for the diffusion kernel')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym',
                        help='normalization for Laplacian')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--batch-norm', action='store_true', help='use batch norm instead of layer norm')
    parser.add_argument('--zero-diag', action='store_true', help='zero diagonal for PE matrix')
    parser.add_argument('--fold-idx', type=int, default=1,
                        help='indices for the train/test datasets')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--test', action='store_true', help='train on full train+val dataset')
    parser.add_argument('--gnn_type', type=str, default='ChebConvDynamic',
                        help='Class name of the gnn to be used in the encoder layer')
    parser.add_argument('--last_layer_filter', action='store_false', help='do not use filter only in the last encoder layer but at every layer')
    parser.add_argument('--learn_only_filter_order_coeff', action='store_true', help='learn only the filter coefficients corresponding to the order and not the projection weights')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

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
        lapdir = 'NoPE' if not args.lappe else 'Lap{}'.format(args.lap_dim) 
        outdir = outdir + '/{}'.format(lapdir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.weight_decay, args.dropout, args.lr, args.nb_layers, args.nb_heads, args.dim_hidden, bn,
            args.pos_enc, args.normalization, args.p, args.beta
        )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/fold-{}'.format(args.fold_idx)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args

def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc

def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for i, (data, mask, pe, lap_pe, degree, labels, edge_indices, batch_indices, feature_indices_to_gather) in enumerate(loader):
        # labels = labels.view(-1)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        # print(optimizer.param_groups[0]['lr'])
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
        output, coeff_reg_term = model(data, edge_indices, batch_indices, feature_indices_to_gather, mask, pe, lap_pe, degree)
        # loss = criterion(output, labels) + coeff_reg_term
        loss = criterion(output, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any():
                    print('== nan')
                    print(name)
                    print(param)
                    import pdb; pdb.set_trace()                    
                if torch.max(torch.abs(param))>1000:
                    print('== max')
                    print(name)
                    print(param)
                    print(torch.max(torch.abs(param)))
                    import pdb; pdb.set_trace()
                if param.grad is not None and torch.isnan(param.grad).any():
                    print('== grad')
                    print(name)
                    print(param)
                    import pdb; pdb.set_trace()
                    nan_params = [name for name, param in model.named_parameters() if param.requires_grad and torch.isnan(param.grad).any() ]

        optimizer.step()

        # pred = output.data.argmax(dim=-1)
        pred = output.data
        running_loss += loss.item() * len(data)
        # running_acc += torch.sum(pred == labels).item()
        running_acc += accuracy_SBM(pred, labels)

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    # epoch_acc = running_acc / n_sample
    epoch_acc = running_acc / (i+1)
    print('Train loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    with torch.no_grad():
        for i, (data, mask, pe, lap_pe, degree, labels, edge_indices, batch_indices, feature_indices_to_gather) in enumerate(loader):
            # labels = labels.view(-1)
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

            output, _ = model(data, edge_indices, batch_indices, feature_indices_to_gather, mask, pe, lap_pe, degree)
            loss = criterion(output, labels)

            # pred = output.data.argmax(dim=-1)
            pred = output.data
            # running_acc += torch.sum(pred == labels).item()
            running_acc += accuracy_SBM(pred, labels)

            running_loss += loss.item() * len(data)
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    # epoch_acc = running_acc / n_sample
    epoch_acc = running_acc / (i+1)
    print('Val loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss

def main():
    global args
    utils.init_device()

    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = 'dataset/SBM'

    dset_name = args.dataset

    train_split = datasets.GNNBenchmarkDataset(data_path, dset_name, split='train')
    val_split = datasets.GNNBenchmarkDataset(data_path, dset_name, split='val')
    test_split = datasets.GNNBenchmarkDataset(data_path, dset_name, split='test')
    n_tags = None#dset[0].num_node_features
    nb_class = train_split.num_classes

    if not os.path.exists("cache/pe/{}".format(args.dataset)):
        try:
            os.makedirs("cache/pe/{}".format(args.dataset))
        except Exception:
            pass


    train_dset = GraphDataset_sbm(train_split, n_tags, degree=True)
    input_size = train_dset.input_size()
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dset.collate_fn())
    print(len(train_dset))
    print(train_dset[0])
    val_dset = GraphDataset_sbm(val_split, n_tags, degree=True)
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
            pos_cache_path = 'cache/pe/{}/{}_{}_{}.pkl'.format(
                args.dataset, args.pos_enc, args.normalization, pos_encoding_params_str)
            pos_encoder = pos_encoding_method(
                pos_cache_path, normalization=args.normalization, zero_diag=args.zero_diag, **pos_encoding_params)

        print("Position encoding...")
        pos_encoder.apply_to(train_dset, split='train')
        pos_encoder.apply_to(val_dset, split='val')
        # train_dset.pe_list = [dset.pe_list[i] for i in train_fold_idx]
        # val_dset.pe_list = [dset.pe_list[i] for i in val_fold_idx]

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
        lap_pos_encoder.apply_to(train_dset)
        lap_pos_encoder.apply_to(val_dset)

    if args.pos_enc is not None:
        model = DiffGraphTransformerGenGCNSBM(in_size=input_size,
                                     nb_class=nb_class,
                                     d_model=args.dim_hidden,
                                     dim_feedforward=2*args.dim_hidden,
                                     dropout=args.dropout,
                                     nb_heads=args.nb_heads,
                                     nb_layers=args.nb_layers,
                                     batch_norm=args.batch_norm,
                                     lap_pos_enc=args.lappe,
                                     lap_pos_enc_dim=args.lap_dim,
                                     gnn_type=args.gnn_type,
                                     last_layer_filter=args.last_layer_filter,
                                     learn_only_filter_order_coeff=args.learn_only_filter_order_coeff)
    else:
        # TODO: implement and import
        model = DiffGraphTransformerGenGCNSBM(in_size=input_size,
                                 nb_class=nb_class,
                                 d_model=args.dim_hidden,
                                 dim_feedforward=2*args.dim_hidden,
                                 dropout=args.dropout,
                                 nb_heads=args.nb_heads,
                                 nb_layers=args.nb_layers,
                                 lap_pos_enc=args.lappe,
                                 lap_pos_enc_dim=args.lap_dim,
                                 gnn_type=args.gnn_type,
                                 learn_only_filter_order_coeff=args.learn_only_filter_order_coeff)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    if nb_class == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    test_dset = GraphDataset_sbm(test_split, n_tags, degree=True)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dset.collate_fn())
    if pos_encoder is not None:
        # test_dset.pe_list = [dset.pe_list[i] for i in test_fold_idx]
        pos_encoder.apply_to(test_dset, split='test')

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder.apply_to(test_dset)

    print("Training...")
    best_val_acc = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    try:
        for epoch in range(args.epochs):
            print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
            train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
            val_acc, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda)
            test_acc, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

            if args.warmup is None:
                lr_scheduler.step()

            logs['train_loss'].append(train_loss)
            logs['val_acc'].append(val_acc)
            logs['val_loss'].append(val_loss)
            logs['test_acc'].append(test_acc)
            logs['test_loss'].append(test_loss)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())

                logs_df = pd.DataFrame.from_dict(logs)
                logs_suffix = '_test' if args.test else ''
                logs_df.to_csv(args.outdir + '/logs{}.csv'.format(logs_suffix))
                results = {
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                }
                results = pd.DataFrame.from_dict(results, orient='index')
                results.to_csv(args.outdir + '/results{}.csv'.format(logs_suffix),
                               header=['value'], index_label='name')
                torch.save(
                    {'args': args,
                    'state_dict': best_weights},
                    args.outdir + '/model{}.pkl'.format(logs_suffix))
                print('Saved!')

        total_time = timer() - start_time
        print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))
        model.load_state_dict(best_weights)

        print()
        print("Testing...")
        test_acc, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

        print("test Acc {:.4f}".format(test_acc))

        # if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs_suffix = '_test' if args.test else ''
        logs.to_csv(args.outdir + '/logs{}.csv'.format(logs_suffix))
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results{}.csv'.format(logs_suffix),
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model{}.pkl'.format(logs_suffix))
        print('Saved!')
    except KeyboardInterrupt as e:
        print('== Exception: ',e)
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model_last_{}.pkl'.format(logs_suffix))    
        logs_df = pd.DataFrame.from_dict(logs)
        logs_df.to_csv(args.outdir + '/logs_last.csv')

if __name__ == "__main__":
    main()
