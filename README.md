# GTPLM-GO
This repo contains the benchmark dataset and key codes of the paper "GTPLM-GO: Enhancing Protein Function Prediction through Dual-branch Graph Transformer and Protein Language Model Fusing Sequence and Local-Global PPI Information"

## Benchmark Dataset

### data

./data is the benchmark dataset from [DeepGraphGO](https://github.com/yourh/DeepGraphGO). It is composed as follows:

- ./data/pid_list.txt: each protein name a line
- ./data/go.txt: each pair of protein name and GO a line
- ./data/ppi_mat.npz: adjacency matrix in scipy.sparse.csr_matrix
- ./data/ppi_interpro.npz: the intepro binary feature of each protein in ppi_pid_list.txt
- ./data/ppi_blastdb: the blastdb of ppi.fasta
- ./data/network_seq.npy: embeddings of protein sequences in the PPI Network based on SeqVec (sum -> 1024 dim)

For more details, please refer to [DeepGraphGO](https://github.com/yourh/DeepGraphGO) and Section 3.1 “Datasets” of the GTPLM-GO paper.

The environment of the code is same with DeepGraphGO.

Benchmark dataset can be downloaded from [here](https://pan.baidu.com/s/1BynGQCdBgu6eo8dU58sdlg?pwd=nntx).

### blast_results

./blast_results is the BLAST sequence alignment results.

blast_results can be downloaded from [here](https://pan.baidu.com/s/1JglJVJ2HwrrjD1flouAPpQ?pwd=qyh7).

## Wandb

The hyperparameter search of GTPLM-GO is based on Wandb, for more information on the use of Wandb, see: [Weights & Biases Docs](https://docs.wandb.ai/)

## Acknowledge

Our code is built upon [DeepGraphGO](https://github.com/yourh/DeepGraphGO), [SGFormer](https://github.com/qitianwu/SGFormer) and [NAGphormer](https://github.com/JHL-HUST/NAGphormer), we thank the authors for their open-sourced code.

For more details one can check the original papers at:

[You, R.; Yao, S.; Mamitsuka, H.; Zhu, S. DeepGraphGO: graph neural network for large-scale, multispecies protein function 559
prediction. Bioinformatics 2021, 37, i262–i271](https://doi.org/10.1093/bioinformatics/btab270)

[Wu, Q.; Zhao, W.; Yang, C.; Zhang, H.; Nie, F.; Jiang, H.; Bian, Y.; Yan, J. SGFormer: Simplifying and Empowering Transformers 581
for Large-Graph Representations. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2023.](https://proceedings.neurips.cc/paper_files/paper/2023/file/cc57fac10eacadb3b72a907ac48f9a98-Paper-Conference.pdf)


## Contact

If you have any questions, please feel free to contact me at [zhanght282018@163.com](mailto:zhanght282018@163.com)

## Cite

If you find this code useful, please consider citing the original work by the authors:
