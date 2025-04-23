import wandb


"""
domain: bp
batch_size: 256
epochs: 1,000
gnn_dropout: 0.4
gnn_num_layers: 2
gnn_weight_decay: 0.00001
hidden_channels： 1,024
lr： 0.00001
patience: 15
seed: 42
seq_weight: 0.8
trans_dropout: 0.3
trans_num_heads: 1
trans_num_layers: 1
trans_weight_decay: 0.00001
"""


"""
domain: mf
batch_size: 256
epochs: 1,000
gnn_dropout: 0.3
gnn_num_layers: 2
gnn_weight_decay: 0
hidden_channels: 1,024
lr: 0.0001
patience: 15
seed: 42
seq_weightL 1.0
trans_dropout: 0.3
trans_num_heads: 1
trans_num_layers: 1
trans_weight_decay: 0.000001
"""


"""
domain: cc
batch_size: 256
epochs: 1,000
gnn_dropout: 0.5
gnn_num_layers: 2
gnn_weight_decay: 0.000001
hidden_channels: 1,024
lr: 0.00005
patience: 15
seed: 42
seq_weight: 0.8
trans_dropout: 0.2
trans_num_heads: 1
trans_num_layers: 1
trans_weight_decay: 0
"""


def get_sweep_configuration(project_name='', args=None, domain='bp'):
    # 超参数搜索方法
    sweep_config = {
        'method': 'bayes'
        # 'method': 'random'
        # 'method': 'grid'
    }
    metric = {
        'name': 'best_fmax_test',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric
    # 参数搜索范围
    parameters_dict = {
        'lr': {
            'values': [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.01]
        },
        'trans_weight_decay': {
            'values': [1e-6, 1e-5, 1e-4, 1e-3, 0]
        },
        'gnn_weight_decay': {
            'values': [1e-6, 1e-5, 1e-4, 1e-3, 0]
        },
        'hidden_channels': {
            'values': [512, 1024]
        },
        # gnn branch
        'gnn_dropout': {
            'values': [0.2, 0.3, 0.4, 0.5, 0.6]
        },
        'gnn_num_layers': {
            'values': [2]
        },
        # trans branch
        'trans_num_heads': {
            'values': [1]
        },
        'trans_num_layers': {
            'values': [1]
        },
        'trans_dropout': {
            'values': [0.2, 0.3, 0.4, 0.5, 0.6]
        },
        'seq_weight': {
            'values': [0.5, 0.6, 0.8, 1.]
        }
    }
    parameters_dict.update({
        'domain': {
            'value': args.domain
        },
        'seed': {
            'value': args.seed
        },
        'batch_size': {
            'value': args.batch_size
        },
        'epochs': {
            'value': args.epochs
        },
        'patience': {
            'value': args.patience
        }
    })
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=project_name + '_' + domain)

    print(sweep_id)

    return sweep_id
