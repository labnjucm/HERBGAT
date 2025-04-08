import scipy
import numpy as np
import pandas as pd
import h5py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.data import DGLDataset

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

def load_hdf_data(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names

def getMetric(parm):
    rows = np.size(parm, axis=0)
    cols = np.size(parm, axis=1)

    y_pred = parm[:, 0]
    # print(f"y_pred")
    # print(parm)
    # print(y_pred)
    #y_pred = np.nan_to_num(y_pred) # updated by cy to handle nan
    y_true = parm[:, 1]
    y_pred_class = y_pred > 0.5
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_class)
    auprc = average_precision_score(y_true, y_pred)

    metric_values = {'auc': auc,
              'auprc': auprc,
              'acc': acc,
              }

    return metric_values

def adjacency2edgelist(adjacency):
    sparse_adj = scipy.sparse.coo_matrix(adjacency)

    edges_src = np.array(sparse_adj.row)
    edges_dst = np.array(sparse_adj.col)

    return edges_src, edges_dst


class PPIDataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name=filename)
        self.filename = filename
        self._num_classes = 2
        self.node_names = None  # 添加这一行

    def process(self):
        network, features, y_trn, y_val, y_tst, trn_mask, val_mask, tst_mask, node_names, feature_names = load_hdf_data(self._name)
        y_trn = y_trn.reshape(-1).astype(int)
        y_val = y_val.reshape(-1).astype(int)
        y_tst = y_tst.reshape(-1).astype(int)
        node_labels = y_trn + y_val + y_tst  

        edges_src, edges_dst = adjacency2edgelist(network)
        node_features = features
        self.node_names = node_names  # 添加这一行
        print("Node Names:", self.node_names)  # 添加这一行，检查 node_names 是否被正确设置

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = torch.FloatTensor(node_features)
        self.graph.ndata['label'] = torch.LongTensor(node_labels)
        self.graph.edata['edges_src'] = torch.LongTensor(edges_src)
        self.graph.edata['edges_dst'] = torch.LongTensor(edges_dst)
        self.graph.ndata['train_mask'] = torch.BoolTensor(trn_mask)
        self.graph.ndata['val_mask'] = torch.BoolTensor(val_mask)
        self.graph.ndata['test_mask'] = torch.BoolTensor(tst_mask)



    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes


class GAT(nn.Module):
    def __init__(self, args, input_dim, hidden_size, num_heads, output_dim=2):
        super(GAT, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(self.args.dropout)

        # graph embedding
        self.gat1 = dgl.nn.GATConv(input_dim, hidden_size, num_heads, 
                                    feat_drop=self.args.dropout, attn_drop=self.args.dropout)
        self.gat2 = dgl.nn.GATConv(hidden_size * num_heads, hidden_size, num_heads, 
                                    feat_drop=self.args.dropout, attn_drop=self.args.dropout)


        self.nonlinear = nn.ReLU()
        self.out_linear = nn.Linear(hidden_size, output_dim)        # init_(nn.Linear(hidden_size, output_dim))

    def forward(self, g, in_feat, action_mask=None, epsilon=1e-9):
        h = self.gat1(g, in_feat).flatten(1)
        h = F.relu(h)
        h = self.gat2(g, h).mean(1)
        emd = h

        logits = self.out_linear(self.nonlinear(emd))
        probs = F.softmax(logits, dim=-1) # masked_softmax(logits, action_mask, dim=-1)

        return probs

    def get_node_probabilities(self, g, in_feat):
        h = self.gat1(g, in_feat).flatten(1)
        h = F.relu(h)
        h = self.gat2(g, h).mean(1)
        emd = h

        logits = self.out_linear(self.nonlinear(emd))
        probs = F.softmax(logits, dim=-1)

        return probs
class GATModel:
    def __init__(self, args, res_path,node_names,device='cpu'):
        self.args = args
        self.res_path = res_path
        self.device = device  #添加这一行
        self.node_names = node_names

        dataset = PPIDataset(args.sample_filename)
        g = dataset[0]
        node_feature = g.ndata['feat']

        self.g = g.to(device)

        self.model = GAT(args, g.ndata['feat'].shape[1], args.hidden_dims, args.heads, args.output_dim).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)

    def get_node_probabilities(self):
        self.model.eval()
        with torch.no_grad():
            features = self.g.ndata['feat'].to(self.device)
            node_probabilities = self.model.get_node_probabilities(self.g, features)

        # 获取节点名称
        node_names = self.node_names

        # 将概率转为 NumPy 数组
        node_probabilities_np = node_probabilities.cpu().numpy()

        # 创建 DataFrame，包含节点名称和概率
        result_df = pd.DataFrame({'NodeName': node_names})
        for i in range(node_probabilities_np.shape[1]):
            result_df[f'Probability_{i}'] = node_probabilities_np[:, i]

        return result_df

    def save_probabilities_to_file(self, result_df, filename='node_probabilities.tsv'):
        # 保存 DataFrame 到 TSV 文件
        result_df.to_csv(os.path.join(self.res_path, filename), sep='\t', index=False)

    def learning(self, mode='train', epsilon=1e-12):
        self.model.train()
        best_auprc_trn = 0
        best_auprc_vld = 0
        best_auprc_tst = 0

        features = self.g.ndata['feat'].to(self.device)
        labels = self.g.ndata['label'].to(self.device)
        labels = torch.stack([1.0-labels.view(-1), labels.view(-1)], dim=1).to(self.device) 
        train_mask = self.g.ndata['train_mask'].to(self.device)
        val_mask = self.g.ndata['val_mask'].to(self.device)
        test_mask = self.g.ndata['test_mask'].to(self.device)
        train_loss1 = []
        train_acc1 = []
        train_auc1 = []
        train_aupr1 = []
        for epoch in range(self.args.num_epochs):
            probs = self.model(self.g, features)

            vec_loss = torch.mul(-torch.log(probs[train_mask] + epsilon), labels[train_mask])
            weighted_vec_loss = torch.mul(vec_loss, torch.tensor([1, self.args.loss_mul]).to(self.device))
            loss = torch.mean(weighted_vec_loss) # weighted cross entrophy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % self.args.print_every == 0:
                y_pred = probs[train_mask].detach().cpu().numpy()
                y_true = labels[train_mask].cpu().numpy()

                parm = np.concatenate([y_pred[:, 1].reshape(-1, 1), y_true[:, 1].reshape(-1, 1)], axis=1)
                metric_values = getMetric(parm)

                avg_loss = float(loss.detach())
                best_flag = ''
                if metric_values['auprc'] > best_auprc_trn:
                    best_auprc_trn = metric_values['auprc']
                    best_flag = 'best'

                print('[trn] epoch:%d, loss=%.4f, auc=%.4f, auprc=%.4f %s, acc=%.4f' %
                    (epoch, avg_loss, metric_values['auc'], metric_values['auprc'], best_flag, metric_values['acc']))
                with open(self.res_path+'/result.txt','a+') as file:
                    file.write('[trn] epoch:%d, loss=%.4f, auc=%.4f, auprc=%.4f %s, acc=%.4f' %
                    (epoch, avg_loss, metric_values['auc'], metric_values['auprc'], best_flag, metric_values['acc'])+'\n')
                train_loss1.append(avg_loss)
                train_acc1.append(metric_values['acc'])
                train_auc1.append(metric_values['auc'])
                train_aupr1.append(metric_values['auprc'])
                self.model.eval()
                best_auprc_vld, best_flag_vld = self.valid_by_batch(epoch, best_auprc_vld, probs[val_mask].cpu(), labels[val_mask].cpu(), mode='vld')
                best_auprc_tst, best_flag_tst = self.valid_by_batch(epoch, best_auprc_tst, probs[test_mask].cpu(), labels[test_mask].cpu(), mode='tst', mark=best_flag_vld)
                self.model.train()
        # with open(os.path.join(self.res_path, 'result_loss.txt'), 'a+') as f:
        #     f.write(str(train_loss1)+'\n')
        # with open(os.path.join(self.res_path, 'result_acc1.txt'), 'a+') as f:
        #     f.write(str(train_acc1)+'\n')
        # with open(os.path.join(self.res_path, 'result_auc1.txt'), 'a+') as f:
        #     f.write(str(train_auc1)+'\n')
        # with open(os.path.join(self.res_path, 'result_aupr1.txt'), 'a+') as f:
        #     f.write(str(train_aupr1)+'\n')

    ####################################
    def get_node_representations(self):
        self.model.eval()
        with torch.no_grad():
            features = self.g.ndata['feat'].to(self.device)
            node_representations = self.model.gat1(self.g, features).mean(1)
        print(node_representations)
        print(node_representations.shape)
        return node_representations

    def valid_by_batch(self, ep, best_metric, probs, labels, mode='vld', epsilon=1e-12, saving=False, mark=None):
        self.model.eval()

        probs = probs.detach()
        y_pred = probs.numpy()
        y_true = labels.numpy()
        vec_loss = torch.mul(-torch.log(probs + epsilon), labels)
        avg_loss = torch.mean(vec_loss)

        parm = np.concatenate([y_pred[:, 1].reshape(-1, 1), y_true[:, 1].reshape(-1, 1)], axis=1)
        metric_values = getMetric(parm)

        best_flag = ''
        if metric_values['auprc'] > best_metric:
            best_metric = metric_values['auprc']
            best_flag = 'best'
        if mark == 'best':
            best_flag = 'select' + best_flag if best_flag == 'best' else 'select'

        print('[%s] epoch:%d, loss=%.4f, auc=%.4f aupr=%.4f %s, acc=%.4f' %
            (mode, ep, avg_loss, metric_values['auc'], metric_values['auprc'], best_flag, metric_values['acc']))
        with open(self.res_path+'/result.txt', 'a+') as file:
            file.write('[%s] epoch:%d, loss=%.4f, auc=%.4f aupr=%.4f %s, acc=%.4f' %
            (mode, ep, avg_loss, metric_values['auc'], metric_values['auprc'], best_flag, metric_values['acc'])+'\n')

        return best_metric, best_flag

