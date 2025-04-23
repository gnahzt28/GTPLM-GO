import os
import random
import argparse
import torch
import dgl
import dgl.data
import numpy as np
import wandb
from sweep_config import get_sweep_configuration
from data_protein import get_graph, get_train_data, get_test_data
from model import GTPLMGO
from train_wandb import Trainer
from pathlib import Path
from logzero import logger


def parser_add_main_args(parser):
    parser.add_argument('--domain', type=str, default='bp')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=15)


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    with wandb.init():
        config = wandb.config

        set_rand_seed(config.seed)

        # ppi 网络和节点的 seqvec 特征
        dgl_graph, network_x, network_seq, graph = get_graph(config.domain)

        # 数据集
        train_ppi, train_y, valid_ppi, valid_y, labels_num = get_train_data(config.domain)
        test_ppi, test_pid_list, test_mlb, test_labels_num, test_res_idx_ = get_test_data(config.domain)

        dataset_data = {
            'dgl_graph': dgl_graph,
            'network_x': network_x,
            'network_seq': network_seq,
            'train_ppi': train_ppi,
            'train_y': train_y,
            'valid_ppi': valid_ppi,
            'valid_y': valid_y,
            'test_ppi': test_ppi,
            'test_pid_list': test_pid_list,
            'test_mlb': test_mlb,
            'test_labels_num': test_labels_num,
            'test_res_idx_': test_res_idx_,
            'test_go_file_path': F'data/{config.domain}_test_go.txt'
        }

        model = GTPLMGO(in_channels=graph['node_feat'].shape[1], hidden_channels=config.hidden_channels, out_channels=labels_num,
                        trans_num_layers=config.trans_num_layers, trans_num_heads=config.trans_num_heads, trans_dropout=config.trans_dropout,
                        gnn_num_layers=config.gnn_num_layers, gnn_dropout=config.gnn_dropout,
                        seq_weight=config.seq_weight)

        trainer = Trainer(data=dataset_data, model=model,
                          epochs=config.epochs, lr=config.lr, patience=config.patience, batch_size=config.batch_size,
                          trans_weight_decay=config.trans_weight_decay, gnn_weight_decay=config.gnn_weight_decay)
        trainer.train()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description='train and validation')
    parser_add_main_args(parser)
    args = parser.parse_args()

    wandb.login()

    # generate a sweep_id
    # sweep_id = get_sweep_configuration(project_name='', args=args, domain='mf')  # domain bp mf cc

    sweep_id = 'your sweep_id'
    wandb.agent(sweep_id=sweep_id, function=main)
