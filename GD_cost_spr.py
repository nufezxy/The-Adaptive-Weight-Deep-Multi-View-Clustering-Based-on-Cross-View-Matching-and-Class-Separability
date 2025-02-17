#计算梯度下降中损失函数的值，用于指导梯度更新。
import torch
def GD_cost_spr(alpha, P, Sm, v, grad, lambda_, gama):
    #alpha学习率，控制梯度更新步长
    #grad梯度向量，大小为f，表示损失函数关于v的偏导
    n_m = Sm.shape[1]  # 样本数量
    # 更新特征权重 v
    v = v - alpha * grad
    # 计算权重的平方
    w = v ** 2

    # 计算逻辑回归模型的预测边界 pm
    p_m = torch.matmul(Sm, w)  # 使用 PyTorch 的矩阵乘法
    Logis = 1 / (1 + torch.exp(-p_m))

    # 防止数值错误
    Logis[Logis == 0] = 1e-10  # 替换为小的正数以避免对数零的情况
    # 计算逻辑回归误差项 f_m
    f_m = (1 / n_m) * torch.sum(torch.log(Logis))

    # 计算正则化项
    if w.ndim > 2:
        # 对于多维张量，使用 permute 来反转维度
        w_transpose = w.permute(*torch.arange(w.ndim - 1, -1, -1))
    else:
        # 对于二维张量，使用 t() 来转置
        w_transpose = w.t()

    #输出f：当前损失函数的值
    f = gama * torch.sum(w) + lambda_ * torch.matmul(w_transpose, torch.matmul(P, w)) - f_m

    # 返回标量值
    return f.item()  # 使用 `.item()` 将张量转换为标量