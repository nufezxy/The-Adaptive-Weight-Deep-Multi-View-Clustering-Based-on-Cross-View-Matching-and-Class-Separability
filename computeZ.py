import torch
import numpy as np

def computeZ(X, id, weight, sigma_margin):
    """
    计算特征差异矩阵 Z_x
    参数:
    X: 特征矩阵 (样本数, 特征维数) -> torch.Tensor
    id: 样本标签向量 (样本数,) -> torch.Tensor
    weight: 特征权重向量 (特征维数,) -> torch.Tensor
    sigma_margin: 高斯核参数，控制距离的衰减速率

    返回:
    Z_x: 特征差异矩阵 (样本数, 特征维数) -> torch.Tensor
    """
    weight = torch.tensor(weight, device=X.device) if isinstance(weight, np.ndarray) else weight
    # Ensure the data type consistency for matmul operation
    weight = weight.float()  # Convert weight to float32
    X = X.float()  # Convert X to float32

    NsRef = X.shape[0]  # 样本数
    f1 = X.shape[1]  # 特征数
    Z_x = torch.zeros((NsRef, f1), device=X.device)  # 初始化差值矩阵 Z_x，大小为 (样本数, 特征数)

    # 遍历每个样本
    for ii in range(NsRef):
        x_ii = X[ii, :]  # 得到当前样本的特征向量（特征维数）
        subj = id[ii]  # 读取当前样本的标签
        ind_P = (id == subj).squeeze()  # 同类样本（正样本）的逻辑索引
        ind_N = (~(id == subj)).squeeze()  # 异类样本（负样本）的逻辑索引

        # 计算当前样本 xi 与所有样本在每个特征维度上的绝对差值。
        Temp = torch.abs(X - x_ii.unsqueeze(0))  # (样本数, 特征维数)
        # 从 Temp 中选择 ind_N 对应的行，构建一个只包含负样本的差异矩阵。
        Temp_N = Temp[ind_N]  # 负样本差异矩阵 (负样本数, 特征维数)

        # 根据权重计算负样本加权距离（公式18）
        if Temp_N.numel() == 0:
            NM = torch.zeros_like(x_ii)  # 无负样本，设置 NM 为零
        else:
            dist = torch.matmul(Temp_N, weight)  # 负样本加权距离
            prob = torch.exp(-dist / sigma_margin)
            if torch.sum(prob) != 0:
                prob_1 = prob / torch.sum(prob)  # 归一化概率
            else:
                prob_1 = torch.zeros_like(prob)
                prob_1[torch.argmin(dist)] = 1  # 最近的负样本概率设为 1
            NM = torch.matmul(prob_1, Temp_N)  # 加权平均特征差异

#*********************************************************************************

        # 计算正样本的加权差异 NH
        ind_P[ii] = False  # 排除当前样本自身
        Temp_P = Temp[ind_P]  # 正样本差异矩阵 (正样本数, 特征维数)
        # 根据权重计算正样本加权距离
        if Temp_P.numel() == 0:
            NH = torch.zeros_like(x_ii)  # 无正样本，设置 NH 为零
        else:
            dist = torch.matmul(Temp_P, weight)  # 正样本加权距离
            prob = torch.exp(-dist / sigma_margin)
            if torch.sum(prob) != 0:
                prob_1 = prob / torch.sum(prob)  # 归一化概率
            else:
                prob_1 = torch.zeros_like(prob)
                prob_1[torch.argmin(dist)] = 1  # 最近的正样本概率设为 1
            NH = torch.matmul(prob_1, Temp_P)  # 加权平均特征差异


        # 差值计算 dx=dnmx-dnhx
        Z_x[ii, :] = NM - NH

    return Z_x
