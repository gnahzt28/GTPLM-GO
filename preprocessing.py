import os
import argparse
import numpy as np
import scipy.sparse as ssp
import networkx as nx
import dgl
import dgl.data
from tqdm import tqdm, trange
from logzero import logger
from pathlib import Path


def parser_add_main_args(parser):
    parser.add_argument('--ppi_net_mat_path', type=str, default='data/ppi_mat.npz')
    parser.add_argument('--dgl_graph_path', type=str, default='data/ppi_dgl_top_100')
    parser.add_argument('--top', type=int, default=100)


def get_norm_net_mat(net_mat):
    degree_0 = np.asarray(net_mat.sum(0)).squeeze()
    mat_d_0 = ssp.diags(degree_0 ** -0.5, format='csr')
    degree_1 = np.asarray(net_mat.sum(1)).squeeze()
    mat_d_1 = ssp.diags(degree_1 ** -0.5, format='csr')
    return mat_d_0 @ net_mat @ mat_d_1


def main():
    parser = argparse.ArgumentParser(description='preprocessing')
    parser_add_main_args(parser)
    args = parser.parse_args()

    ppi_net_mat_path = Path(args.ppi_net_mat_path)
    dgl_graph_path = args.dgl_graph_path
    top = args.top

    """
    ssp.load_npz 加载了 ppi 网络的稀疏邻接矩阵，并赋值给 mat_
    ssp.eye 创建了一个稀疏单位矩阵，其大小与 mat_ 的行数相同，其与 mat_ 相加获得 ppi_net_mat，相当于加自环
    """
    ppi_net_mat = (mat_ := ssp.load_npz(ppi_net_mat_path)) + ssp.eye(mat_.shape[0], format='csr')
    # ppi_net_mat.shape 矩阵的形状（即行数和列数），ppi_net_mat.nnz 矩阵中非零元素的数量
    logger.info(F'{ppi_net_mat.shape} {ppi_net_mat.nnz}')
    r, c, v = [], [], []
    # 外层循环遍历稀疏矩阵的每一行（即每个节点），trange 是 tqdm 库的函数，用于显示循环的进度条
    for i in trange(ppi_net_mat.shape[0]):
        """
        内层循环对每个节点的连接进行排序，并只保留最强的 top 个连接
        这是通过 sorted(zip(ppi_net_mat[i].data, ppi_net_mat[i].indices), reverse=True)[:top] 实现的
        首先将非零元素值（.data）和它们的列索引（.indices）打包成元组，然后按值降序排序，并最后只取前top个元素
        """
        for v_, c_ in sorted(zip(ppi_net_mat[i].data, ppi_net_mat[i].indices), reverse=True)[:top]:
            # 将当前节点的索引 i、最强连接的列索引 c_ 和对应的值 v_ 分别添加到列表 r、c 和 v 中
            r.append(i)
            c.append(c_)
            v.append(v_)
    """
    使用 scipy.sparse 的 csc_matrix 函数创建了一个压缩稀疏列（CSC）格式的矩阵
    CSC 格式适用于按列进行操作的稀疏矩阵，输入参数(v, (r, c))分别表示非零元素的值、行索引和列索引，这些都是从之前的循环中获得的
    shape=ppi_net_mat.shape 确保了新矩阵与原始矩阵 ppi_net_mat 具有相同的形状
    .T 是转置操作，将 CSC 矩阵转换为 CSR 格式
    get_norm_net_mat(...) 用来对稀疏矩阵进行归一化
    """
    ppi_net_mat = get_norm_net_mat(ssp.csc_matrix((v, (r, c)), shape=ppi_net_mat.shape).T)
    # 归一化后的稀疏矩阵 ppi_net_mat 的形状和非零元素的数量
    logger.info(F'{ppi_net_mat.shape} {ppi_net_mat.nnz}')
    """
    使用 scipy.sparse 的 coo_matrix 函数将稀疏矩阵转换为 COO 格式
    COO 格式存储非零元素的值以及它们的行索引和列索引，这使得迭代非零元素变得容易
    """
    ppi_net_mat_coo = ssp.coo_matrix(ppi_net_mat)
    """
    使用 networkx 库创建一个有向图（DiGraph）
    通过迭代 COO 格式稀疏矩阵的非零元素，将有向边添加到 NetworkX 图中，每条边都带有一个属性 ppi，其值等于稀疏矩阵中的非零元素值
    """
    nx_graph = nx.DiGraph()
    for u, v, d in tqdm(zip(ppi_net_mat_coo.row, ppi_net_mat_coo.col, ppi_net_mat_coo.data),
                        total=ppi_net_mat_coo.nnz, desc='PPI'):
        nx_graph.add_edge(u, v, ppi=d)
    # 使用 DGL 库的 from_networkx 方法将 NetworkX 图转换为 DGL 图，同时指定了要保留的边属性 ppi
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph, edge_attrs=('ppi',))
    assert dgl_graph.in_degrees().max() <= top
    # 使用 DGL 库的 save_graphs 函数将 DGL 图保存到指定的路径 dgl_graph_path，后续可以直接加载图而不需要重新计算
    dgl.data.utils.save_graphs(dgl_graph_path, dgl_graph)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
