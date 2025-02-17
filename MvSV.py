import numpy as np
import torch
from computeZ import computeZ
from GD_solver_spr import GD_solver_spr

def MvSV(X, Y,id, sigma, lambda_, gama):
    itr_max = 500  # 最大迭代次数
    f1 = X.shape[1]  # 第一视图的特征数 20
    f2 = Y.shape[1]  # 第二视图的特征数 20
    NRef = len(id) - 1  # 样本数量减一（参考样本数量）
    w = np.ones(f1 + f2) / np.sqrt(f1 + f2)  # 初始化联合权重向量，长度为f1+f2——论文里的fx
    w1 = w[:f1]  # 初始化第一视图权重
    w2 = w[f1:]  # 初始化第二视图权重

    Difference = 1  # 初始化权重变化的差值
    t = 0  # 迭代次数
    w_old = w  # 保存上一轮的权重

    while Difference > 0.01 and t <= 4:
        t += 1
        # 计算差值矩阵，论文里的dx，dy
        Z_x = computeZ(X, id,w1, sigma)  # 计算第一视图的差值 256,20
        Z_y = computeZ(Y, id,w2, sigma)  # 计算第二视图的差值 256,20

        #构建联合矩阵（公式24）
        top = torch.cat([Z_x, torch.zeros((Z_x.shape[0], Z_y.shape[1]), device=Z_x.device)], dim=1)
        bottom = torch.cat([torch.zeros((Z_y.shape[0], Z_x.shape[1]), device=Z_x.device), Z_y], dim=1)
        Sm = torch.cat([top, bottom], dim=0) #512,40

        # 确保 Sm 是 PyTorch 张量
        if isinstance(Sm, np.ndarray):
            Sm = torch.tensor(Sm, dtype=torch.float32, device=X.device)  # 转换为 PyTorch 张量，确保在同一设备上


#**********************************************************************************************************************
        # 最小化跨视图匹配误差：P对应论文中的D
        P = torch.zeros((f1 + f2, f1 + f2), device=X.device)  # 初始化矩阵，确保在相同的设备上
        for ss in range(NRef):
            ind_mN = list(range(NRef))
            ind_mN.remove(ss)  # 去掉当前样本的索引
            # 计算样本之间的差异
            Dxi = torch.abs(X[ind_mN, :] - X[ss, :])  # 第一视图样本差异
            Dyi = torch.abs(Y[ind_mN, :] - Y[ss, :])  # 第二视图样本差异
            # 计算正则化矩阵
            P += torch.cat([
                torch.cat([Dxi.T @ Dxi, -Dxi.T @ Dyi], dim=1),
                torch.cat([-Dyi.T @ Dxi, Dyi.T @ Dyi], dim=1)
            ], dim=0)
        P /= NRef  # 平均化差异矩阵D


# **********************************************************************************************************************
        # 梯度下降优化
        w, fval = GD_solver_spr(w, P, Sm, lambda_, gama, itr_max)

        # 更新权重并检查变化
        w1 = w[:f1]  # 更新第一视图的权重
        w2 = w[f1:]  # 更新第二视图的权重

        # 计算损失值
        f = torch.cat([w1, w2], dim=0)  # 在第 0 维度上连接张量
        # 确保 w 和 w_old 在相同的设备上 (例如，GPU 上)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w = w.to(device)
        w_old = torch.tensor(w_old, device=device)

        # 计算权重变化的范数
        Difference = torch.norm(torch.abs(w / torch.max(w)) - w_old / torch.max(w_old))  # 权重变化的范数
        w_old = w  # 更新历史权重

    # 返回f，Sm，P
    return f, Sm, P,w1,w2
