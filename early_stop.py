import copy
import operator
import numpy as np
from typing import List
from enum import Enum, auto
from torch.nn import Module


# --------------------------------------------------------
# Codes are borrowed from NAGphormer
# --------------------------------------------------------


class StopVariable(Enum):
    LOSS = auto()
    ACCURACY = auto()


class Best(Enum):
    # 只要有任何一项指标变优且与历史最佳不同，就保存当前模型状态
    RANKED = auto()
    # 当所有指标均优于历史最佳时，保存当前模型状态
    ALL = auto()


def Stop_args(patience=15, max_epochs=1000, stop_varnames=None):
    if stop_varnames is None:
        stop_varnames = [StopVariable.ACCURACY, StopVariable.ACCURACY, StopVariable.LOSS]
    return dict(stop_varnames=stop_varnames, patience=patience, max_epochs=max_epochs, remember=Best.RANKED)


class EarlyStopping:
    def __init__(self, model: Module, stop_varnames: List[StopVariable], patience=15, max_epochs=1000,
                 remember=Best.ALL):
        self.model = model
        # 运算符列表，根据停止变量名来选择相应的比较运算符，即 loss 对应 <=，acc 对应 >=
        self.comp_ops = []
        # 停止变量名列表，如 loss 或 acc
        self.stop_vars = []
        # 最佳值列表
        self.best_vals = []
        # 遍历停止变量名，根据值的类型做初始化
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                # 前指标需要小于等于历史最优（loss 越小越好），初始化损失的最佳值为正无穷大
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                # 当前指标需要大于等于历史最优（accuracy 越大越好），初始化准确率的最佳值为负无穷大
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        # 记录保存最优模型状态的方式 Best.ALL 或者 Best.RANKED
        self.remember = remember
        # 记住的历史最优值列表
        self.remembered_vals = copy.copy(self.best_vals)
        # 最大耐心值
        self.max_patience = patience
        # 当前耐心值（计数器），初始化为最大耐心值
        self.patience = self.max_patience
        # 存储最佳模型对应的训练轮数
        self.max_epochs = max_epochs
        # 记录最优模型的 epoch
        self.best_epoch = None
        # 存储最佳模型的状态
        self.best_state = None

    def check(self, values: List[np.floating], epoch: int) -> bool:
        """
        用于检查当前 epoch 的指标值是否优于之前记录的最佳值，并更新最佳值、耐心计数器以及最优模型状态
        :param values: 一个数值列表，包含模型的一些评价指标
        :param epoch: 当前的训练轮次
        :return:
        """

        # 对于每个指标，检查当前值是否优于历史最优值，其中元素是布尔值，表示当前指标值是否优于历史最佳值
        checks = [self.comp_ops[i](val, self.best_vals[i]) for i, val in enumerate(values)]
        # 如果有任何一个指标优于其历史最佳值，则执行以下操作
        if any(checks):
            # 使用 numpy.choose 函数根据 checks 列表中的布尔值从 [self.best_vals, values] 中选择数组元素，
            # 即如果 checks[i] 为真，则取 values[i] 作为新的最佳值，否则保持原最佳值不变
            self.best_vals = np.choose(checks, [self.best_vals, values])
            # 将耐心计数器重置为其最大值
            self.patience = self.max_patience
            # 布尔值列表，这个新列表的每个元素表示当前指标值（从 values 列表中获取）是否优于之前记住的历史最优值（从 self.remembered_vals 列表中获取）
            comp_remembered = [self.comp_ops[i](val, self.remembered_vals[i]) for i, val in enumerate(values)]
            # 如果 remember 设置为 Best.ALL，并且当前所有指标都优于之前记住的历史最优值
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    # 更新当前最优 epoch 和最优指标值
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    # 获取并存储模型当前的状态字典（即将模型参数从 GPU 转移到 CPU）
                    self.best_state = {
                        key: value.cpu() for key, value
                        in self.model.state_dict().items()}
            # 如果 remember 设置为 Best.RANKED
            elif self.remember is Best.RANKED:
                # 遍历每个指标比较结果，检查它是否比之前记住的最好值要好
                for i, comp in enumerate(comp_remembered):
                    # 当前指标优于已记住的历史最优值时
                    if comp:
                        # 如果当前值与记住的历史最优值不相等（防止相同数值导致重复保存）
                        if not (self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            # 更新当前最优 epoch、最优指标值以及模型状态
                            self.best_state = {
                                key: value.cpu() for key, value
                                in self.model.state_dict().items()}
                            # 找到一个变优的指标后立即跳出循环
                            break
                    else:
                        break
        else:
            # 如果没有任何指标优于其历史最佳值，则减少一次耐心计数
            self.patience -= 1
        # 返回当前耐心计数器是否已经归零，若为 True 表示满足早停条件，应停止训练
        return self.patience == 0
