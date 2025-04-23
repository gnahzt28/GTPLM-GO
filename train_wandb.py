import torch
import dgl
import numpy as np
import wandb
from tqdm import tqdm
from torch import nn
from model import GTPLMGO
from logzero import logger
from itertools import chain
from early_stop import StopVariable, Stop_args, EarlyStopping
from data_protein import output_res, get_pid_go_score, get_pid_go_mat, get_pid_go_score_mat, get_pid_go
from metrics import fmax, aupr, ROOT_GO_TERMS


class Trainer:
    def __init__(self, model: GTPLMGO, data, epochs, lr, patience, batch_size, trans_weight_decay, gnn_weight_decay):
        # data
        self.dgl_graph = data['dgl_graph']
        self.network_x = data['network_x']
        self.network_seq = data['network_seq']
        self.train_ppi = data['train_ppi']
        self.train_y = data['train_y']
        self.valid_ppi = data['valid_ppi']
        self.valid_y = data['valid_y']
        self.test_ppi = data['test_ppi']
        self.test_pid_list = data['test_pid_list']
        self.test_mlb = data['test_mlb']
        self.test_labels_num = data['test_labels_num']
        self.test_res_idx_ = data['test_res_idx_']
        self.test_go_file_path = data['test_go_file_path']
        # params
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.trans_weight_decay = trans_weight_decay
        self.gnn_weight_decay = gnn_weight_decay
        # model
        self.model = model
        self.model.cuda()
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.reset_parameters()

        self.optimizer = torch.optim.Adam(
            [
                {'params': self.model.params1, 'weight_decay': self.trans_weight_decay},
                {'params': self.model.params2, 'weight_decay': self.gnn_weight_decay}
            ],
            lr=self.lr
        )

    def train(self):
        # early stop
        stop_args = Stop_args(patience=self.patience, max_epochs=self.epochs, stop_varnames=[StopVariable.LOSS])
        early_stopping = EarlyStopping(self.model, **stop_args)
        valid_eval_list = []
        test_eval_list = []

        ppi_train_idx = np.full(self.network_x.shape[0], -1, dtype=np.int)
        ppi_train_idx[self.train_ppi] = np.arange(self.train_ppi.shape[0])

        ppi_valid_idx = np.full(self.network_x.shape[0], -1, dtype=np.int)
        ppi_valid_idx[self.valid_ppi] = np.arange(self.valid_ppi.shape[0])

        ppi_test_idx = np.full(self.network_x.shape[0], -1, dtype=np.int)
        ppi_test_idx[self.test_ppi] = np.arange(self.test_ppi.shape[0])

        pid_go = get_pid_go(self.test_go_file_path)
        pid_list = list(pid_go.keys())

        for epoch in range(self.epochs):
            # train
            loss_train = 0.0
            # 根据 batch 采样子图，完成训练
            for nf in tqdm(dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, self.batch_size,
                                                                        self.dgl_graph.number_of_nodes(),
                                                                        num_hops=self.model.gnn_num_layers,
                                                                        seed_nodes=self.train_ppi,
                                                                        prefetch=True, shuffle=True),
                           desc=F'Epoch-{epoch}, Train: ', leave=False, dynamic_ncols=True,
                           total=(len(self.train_ppi) + self.batch_size - 1) // self.batch_size):
                batch_y = self.train_y[ppi_train_idx[nf.layer_parent_nid(-1).numpy()]].toarray()

                self.model.train()
                batch_scores = self.get_batch_scores(nf=nf)
                loss = self.loss_fn(batch_scores, torch.from_numpy(batch_y).cuda())
                loss.backward()
                self.optimizer.step(closure=None)
                self.optimizer.zero_grad()
                loss_train += loss.item()

            # valid
            with torch.no_grad():
                loss_valid = 0.0
                valid_ppi = self.valid_ppi
                for nf in tqdm(dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, self.batch_size,
                                                                            self.dgl_graph.number_of_nodes(),
                                                                            num_hops=self.model.gnn_num_layers,
                                                                            seed_nodes=np.unique(valid_ppi),
                                                                            prefetch=True),
                               desc=F'Epoch-{epoch}, Valid: ', leave=False, dynamic_ncols=True,
                               total=(len(self.valid_ppi) + self.batch_size - 1) // self.batch_size):
                    batch_y = self.valid_y[ppi_valid_idx[nf.layer_parent_nid(-1).numpy()]].toarray()
                    self.model.eval()
                    batch_scores = self.get_batch_scores(nf=nf)
                    loss = self.loss_fn(batch_scores, torch.from_numpy(batch_y).cuda())
                    loss_valid += loss.item()

            with torch.no_grad():
                test_ppi = self.test_ppi
                unique_test_ppi = np.unique(test_ppi)
                mapping = {x: i for i, x in enumerate(unique_test_ppi)}
                test_ppi = np.asarray([mapping[x] for x in test_ppi])
                test_scores_list = []
                for nf in tqdm(dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, self.batch_size,
                                                                            self.dgl_graph.number_of_nodes(),
                                                                            num_hops=self.model.gnn_num_layers,
                                                                            seed_nodes=unique_test_ppi,
                                                                            prefetch=True),
                               desc=F'Epoch-{epoch}, Test: ', leave=False, dynamic_ncols=True,
                               total=(len(self.test_ppi) + self.batch_size - 1) // self.batch_size):
                    self.model.eval()
                    batch_scores = self.get_batch_scores(nf=nf)
                    test_scores_list.append(torch.sigmoid(batch_scores))

                test_scores = torch.cat(test_scores_list, dim=0)
                test_scores = test_scores[test_ppi].cpu().numpy()

                scores = np.zeros((len(self.test_pid_list), self.test_labels_num))
                scores[self.test_res_idx_] = test_scores

                pid_go_score = get_pid_go_score(output_res(self.test_pid_list, self.test_mlb.classes_, scores))

                go_list = sorted(set(list(chain(*([pid_go[p_] for p_ in pid_list] +
                                                  [pid_go_score[p_] for p_ in pid_list if
                                                   p_ in pid_go_score])))) - ROOT_GO_TERMS)
                targets, scores = get_pid_go_mat(pid_go, pid_list, go_list), get_pid_go_score_mat(pid_go_score,
                                                                                                  pid_list, go_list)

                fmax_test = fmax(targets, scores)
                aupr_test = aupr(targets, scores)

            logger.info(F'Epoch-{epoch}')
            logger.info(F'Train: Train Loss={loss_train:.5f}')
            logger.info(F'Valid: Valid Loss={loss_valid:.5f}')
            logger.info(F'Test: Fmax={fmax_test:.3f}, AUPR={aupr_test:.3f}')

            wandb.log(
                {
                    "epoch": epoch,
                    "loss_train": loss_train,
                    "loss_val": loss_valid,
                    "fmax_test": fmax_test,
                    "aupr_test": aupr_test
                }
            )

            valid_eval_list.append({
                'loss': loss_valid
            })

            test_eval_list.append({
                'fmax': fmax_test,
                'aupr': aupr_test
            })

            if early_stopping.check([loss_valid], epoch):
                break

        best_test_metrics = test_eval_list[int(early_stopping.best_epoch)]
        best_valid_metrics = valid_eval_list[int(early_stopping.best_epoch)]

        logger.info(F'Best Metrics:')
        logger.info(F'best_epoch: {early_stopping.best_epoch}')
        logger.info(F'best_loss_valid: {best_valid_metrics["loss"]:.5f}')
        logger.info(F'best_fmax_test: {best_test_metrics["fmax"]:.3f}')
        logger.info(F'best_aupr_test: {best_test_metrics["aupr"]:.3f}')

        wandb.log(
            {
                "best_epoch": early_stopping.best_epoch,
                "best_loss_valid": best_valid_metrics["loss"],
                "best_fmax_test": best_test_metrics["fmax"],
                "best_aupr_test": best_test_metrics["aupr"]
            }
        )

        # save model
        logger.info(F'best model to save: {early_stopping.best_epoch}')
        self.model.load_state_dict(early_stopping.best_state)
        model_save_path = F'/your_save_path/{self.domain}/GTPLM-GO-best-{self.domain}-' \
                          F'fmax-{best_test_metrics["fmax"]}-aupr-{best_test_metrics["aupr"]}.pt'
        logger.info(F'model save path: {model_save_path}')
        torch.save(self.model.state_dict(), model_save_path)

    def get_batch_scores(self, nf: dgl.NodeFlow):
        # batch_x
        batch_x = self.network_x[nf.layer_parent_nid(-1).numpy()]
        batch_x = torch.from_numpy(batch_x.toarray()).cuda().float()

        # batch_seq
        batch_seq = self.network_seq[nf.layer_parent_nid(-1).numpy()]
        batch_seq = torch.from_numpy(batch_seq).cuda().float()

        # batch_neighbor_x
        batch_neighbor_x = self.network_x[nf.layer_parent_nid(0).numpy()]
        batch_neighbor_x = (torch.from_numpy(batch_neighbor_x.indices).cuda().long(),
                            torch.from_numpy(batch_neighbor_x.indptr).cuda().long(),
                            torch.from_numpy(batch_neighbor_x.data).cuda().float())

        return self.model(nf, batch_x=batch_x, batch_seq=batch_seq, batch_neighbor_x=batch_neighbor_x)
