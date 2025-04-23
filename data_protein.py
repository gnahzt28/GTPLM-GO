import os
import sys
import warnings
import numpy as np
import scipy.sparse as ssp
import joblib
import torch
import dgl.data
from logzero import logger
from collections import defaultdict
from pathlib import Path
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio.Blast import NCBIXML
from tqdm import tqdm


# --------------------------------------------------------
# Codes are borrowed from DeepGraphGO
# --------------------------------------------------------


def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file


def get_go_list(pid_go_file, pid_list):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split())[0]].append(line_list[1])
        return [pid_go[pid_] for pid_ in pid_list]
    else:
        return None


def get_data(fasta_file, pid_go_file=None):
    pid_list, data_x = [], []

    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)
        data_x.append(str(seq.seq))

    return pid_list, data_x, get_go_list(pid_go_file, pid_list)


def get_mlb(mlb_path: Path, labels=None) -> MultiLabelBinarizer:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if mlb_path.exists():
            return joblib.load(mlb_path)
        mlb = MultiLabelBinarizer(sparse_output=True)
        mlb.fit(labels)
        joblib.dump(mlb, mlb_path)
        return mlb


def get_ppi_idx(pid_list, data_y, net_pid_map):
    pid_list_ = tuple(zip(*[(i, pid, net_pid_map[pid])
                            for i, pid in enumerate(pid_list) if pid in net_pid_map]))
    assert pid_list_
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def get_homo_ppi_idx(pid_list, fasta_file, data_y, net_pid_map, net_blastdb, blast_output_path):
    blast_sim = blast(net_blastdb, pid_list, fasta_file, Path(blast_output_path))
    pid_list_ = []
    for i, pid in enumerate(pid_list):
        blast_sim[pid][None] = float('-inf')
        pid_ = pid if pid in net_pid_map else max(blast_sim[pid].items(), key=lambda x: x[1])[0]
        if pid_ is not None:
            pid_list_.append((i, pid, net_pid_map[pid_]))
    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def output_res(pid_list, go_list, sc_mat):
    pid_go_score = {}
    for pid_, sc_ in zip(pid_list, sc_mat):
        pid_go_score[pid_] = {}
        for go_, s_ in zip(go_list, sc_):
            if s_ > 0.0:
                pid_go_score[pid_][go_] = s_
    return pid_go_score


def get_pid_go(pid_go_file):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split('\t'))[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None


def get_pid_go_score(pid_go_score: dict):
    res = defaultdict(dict)
    for pid, go_score in pid_go_score.items():
        for go, score in go_score.items():
            res[pid][go] = float(score)
    return dict(res)


def get_pid_go_mat(pid_go, pid_list, go_list):
    go_mapping = {go_: i for i, go_ in enumerate(go_list)}
    r_, c_, d_ = [], [], []
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go:
            for go_ in pid_go[pid_]:
                if go_ in go_mapping:
                    r_.append(i)
                    c_.append(go_mapping[go_])
                    d_.append(1)
    return ssp.csr_matrix((d_, (r_, c_)), shape=(len(pid_list), len(go_list)))


def get_pid_go_score_mat(pid_go_score, pid_list, go_list):
    sc_mat = np.zeros((len(pid_list), len(go_list)))
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go_score:
            for j, go_ in enumerate(go_list):
                sc_mat[i, j] = pid_go_score[pid_].get(go_, -1e100)
    return sc_mat


def get_graph(domain, data_rootpath='data', result_rootpath='blast_results'):
    protein_path_dict = {
        'domain_name': domain,
        'data_rootpath': data_rootpath,  # 默认是 data
        'result_rootpath': result_rootpath,  # 默认是 result
        'dgl_path': F'{data_rootpath}/ppi_dgl_top_100',
        'feature_path': F'{data_rootpath}/ppi_interpro.npz',
        'seq_emb_path': F'{data_rootpath}/network_seq.npy'
    }
    # ppi 网络，节点数量 189065，边数量 17564203
    dgl_graph = dgl.data.utils.load_graphs(protein_path_dict['dgl_path'])[0][0]

    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_:=np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0

    dgl_graph.edata['self'] = self_loop

    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float().cuda()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float().cuda()

    # ppi 的节点的特征向量矩阵 189065(蛋白质数量)×41311(特征数)
    network_x = ssp.load_npz(protein_path_dict['feature_path'])

    # ppi 节点的 SeqVec 特征向量 (1024)
    network_seq = np.load(protein_path_dict['seq_emb_path'])

    graph = {
        'edge_index': torch.stack(dgl_graph.edges()),  # 边索引
        'edge_feat': dgl_graph.edata['ppi'].view(-1, 1),  # 边特征
        'node_feat': network_x,  # 节点特征
        'node_seq_feat': network_seq,  # 节点序列的特征
        'num_nodes': dgl_graph.batch_num_nodes[0]  # 节点数量
    }

    # 图的统计信息
    logger.info(F'Number of Nodes: {graph["num_nodes"]}')
    logger.info(F'Number of Edges: {graph["edge_index"].shape[1]}')
    logger.info(F'Number of Node Features: {graph["node_feat"].shape[1]}\n')

    return dgl_graph, network_x, network_seq, graph


def get_train_data(domain, data_rootpath='data', result_rootpath='blast_results'):
    protein_path_dict = {
        'domain_name': domain,
        'data_rootpath': data_rootpath,  # 默认是 data
        'result_rootpath': result_rootpath,  # 默认是 result
        'mlb_path': F'{data_rootpath}/{domain}_go.mlb',
        'pid_list_path': F'{data_rootpath}/ppi_pid_list.txt',
        'blastdb_path': F'{data_rootpath}/ppi_blastdb',
        # train 相关
        'train_pid_list_file_path': F'{data_rootpath}/{domain}_train_pid_list.txt',
        'train_fasta_file_path': F'{data_rootpath}/{domain}_train.fasta',
        'train_pid_go_file_path': F'{data_rootpath}/{domain}_train_go.txt',
        # valid 相关
        'valid_pid_list_file_path': F'{data_rootpath}/{domain}_valid_pid_list.txt',
        'valid_fasta_file_path': F'{data_rootpath}/{domain}_valid.fasta',
        'valid_pid_go_file_path': F'{data_rootpath}/{domain}_valid_go.txt',
        'valid_blast_output_path': F'{result_rootpath}/{domain}-valid-ppi-blast-out'
    }
    net_pid_list = get_pid_list(pid_list_file=protein_path_dict['pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    net_blastdb = protein_path_dict['blastdb_path']

    train_pid_list, _, train_go = get_data(fasta_file=protein_path_dict['train_fasta_file_path'],
                                           pid_go_file=protein_path_dict['train_pid_go_file_path'])
    valid_pid_list, _, valid_go = get_data(fasta_file=protein_path_dict['valid_fasta_file_path'],
                                           pid_go_file=protein_path_dict['valid_pid_go_file_path'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mlb = get_mlb(Path(protein_path_dict['mlb_path']), train_go)
        labels_num = len(mlb.classes_)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y, valid_y = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)

    *_, train_ppi, train_y = get_ppi_idx(pid_list=train_pid_list, data_y=train_y, net_pid_map=net_pid_map)

    *_, valid_ppi, valid_y = get_homo_ppi_idx(pid_list=valid_pid_list,
                                              fasta_file=protein_path_dict['valid_fasta_file_path'],
                                              data_y=valid_y, net_pid_map=net_pid_map, net_blastdb=net_blastdb,
                                              blast_output_path=protein_path_dict['valid_blast_output_path'])

    # 训练集和验证集的统计信息
    logger.info(F'Number of Labels: {labels_num}')
    logger.info(F'Size of Training Set: {len(train_ppi)}')
    logger.info(F'Size of Validation Set: {len(valid_ppi)}\n')

    return train_ppi, train_y, valid_ppi, valid_y, labels_num


def get_test_data(domain, data_rootpath='data', result_rootpath='blast_results'):
    protein_path_dict = {
        'domain_name': domain,
        'data_rootpath': data_rootpath,  # 默认是 data
        'result_rootpath': result_rootpath,  # 默认是 result
        'mlb_path': F'{data_rootpath}/{domain}_go.mlb',
        'pid_list_path': F'{data_rootpath}/ppi_pid_list.txt',
        'blastdb_path': F'{data_rootpath}/ppi_blastdb',
        # test 相关
        'test_pid_list_file_path': F'{data_rootpath}/{domain}_test_pid_list.txt',
        'test_fasta_file_path': F'{data_rootpath}/{domain}_test.fasta',
        'test_blast_output_path': F'{result_rootpath}/{domain}-test-ppi-blast-out'
    }
    net_pid_list = get_pid_list(pid_list_file=protein_path_dict['pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    net_blastdb = protein_path_dict['blastdb_path']

    test_pid_list, _, test_go = get_data(fasta_file=protein_path_dict['test_fasta_file_path'])

    test_res_idx_, test_pid_list_, test_ppi, _ = \
        get_homo_ppi_idx(pid_list=test_pid_list,
                         fasta_file=protein_path_dict['test_fasta_file_path'],
                         data_y=None, net_pid_map=net_pid_map,
                         net_blastdb=net_blastdb,
                         blast_output_path=protein_path_dict['test_blast_output_path'])

    test_mlb = get_mlb(Path(protein_path_dict['mlb_path']))
    test_labels_num = len(test_mlb.classes_)

    logger.info(F'Number of Test Set Labels: {test_labels_num}')
    logger.info(F'Size of Test Set: {len(test_ppi)}\n')

    return test_ppi, test_pid_list, test_mlb, test_labels_num, test_res_idx_


def psiblast(blastdb, pid_list, fasta_path, output_path: Path, evalue=1e-3, num_iterations=3,
             num_threads=40, bits=True, query_self=False):
    output_path = output_path.with_suffix('.xml')
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cline = NcbipsiblastCommandline(query=fasta_path, db=blastdb, evalue=evalue, outfmt=5, out=output_path,
                                        num_iterations=num_iterations, num_threads=num_threads)
        print(cline)
        cline()

    with open(output_path) as fp:
        psiblast_sim = defaultdict(dict)
        for pid, rec in zip(tqdm(pid_list, desc='Parsing PsiBlast blast_results', file=sys.stdout), NCBIXML.parse(fp)):
            query_pid, sim = rec.query, []
            assert pid == query_pid
            for alignment in rec.alignments:
                alignment_pid = alignment.hit_def.split()[0]
                if alignment_pid != query_pid or query_self:
                    psiblast_sim[query_pid][alignment_pid] = max(
                            hsp.bits if bits else hsp.identities / rec.query_length for hsp in alignment.hsps)
    return psiblast_sim


def blast(*args, **kwargs):
    return psiblast(*args, **kwargs, num_iterations=1)
