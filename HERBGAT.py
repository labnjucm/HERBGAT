from datetime import datetime
import sys
import os

import argparse
import random

import h5py
import numpy as np
import torch
import dgl
from GAT import GATModel
import torch.nn.functional as F
import pandas as pd
import GAT as GT

# python dgl_gat_main.py --num_epochs=1000 --hidden_dims=64 --heads=4 --dropout=0.2 --loss_mul=1 --sample_filename=/home/njucm/zdf/emogi-reusability-main/essential_gene/CPDB_essential_test01_multiomics.h5 --lr=0.001 --seed=1 --cuda

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random generator seed.', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--hidden_dims', type=int, default=128, help='hidden dims for head')
    parser.add_argument('--loss_mul', type=int, default=1)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument("--print_every", type=int, default=10, help='eval_interval')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--heads', type=int, default=4, help='Num of heads for GAT layer')
    parser.add_argument('--sample_filename', type=str, default='/research/dept4/cyhong/cancer_gcn/EMOGI/saved_model/EMOGI_CPDB/CPDB_multiomics.h5')
    parser.add_argument('--cuda', default=False, action="store_true")

    args = parser.parse_args()
    return args


def set_seed(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    dgl.random.seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return device

def result_Path():
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    root_dir =date
    root_dir = '/home/njucm/zdf/emogi-reusability-main/Result/' + date
    os.mkdir(root_dir)
    return root_dir

if __name__ == '__main__':
    args = get_args()
    device = set_seed(args)

    res_path = result_Path()
    with h5py.File(args.sample_filename, 'r') as f:
        node_names = f['gene_names'][:]
    model = GATModel(args, res_path, node_names=node_names,device=device)
    model.learning()
    with open(res_path+'/result.txt','a+') as file:
        file.write("end time is {}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        # 获取节点表示
    # node_representations = model.get_node_representations()

    node_probabilities_df = model.get_node_probabilities()
    model.save_probabilities_to_file(node_probabilities_df)


    # ranked_nodes,ranked_node_names = model.rank_nodes_by_similarity(node_representations)
    #
    # # 打印排序结果
    # # print("Ranked Nodes:")
    # # for node, node_name in zip(ranked_nodes, ranked_node_names):
    # #     print(f"{node_name}: {node}")
    # output_filename = res_path+'/ranked_nodes.txt'
    # with open(output_filename, 'w') as output_file:
    #     for node, node_name in zip(ranked_nodes, ranked_node_names):
    #         output_file.write(f"{node_name}: {node}\n")