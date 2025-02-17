#实现梯度下降优化，调用 GD_cost_spr.m 计算损失，通过迭代找到特征权重 𝑤，这是优化过程的核心部分。
import torch
import numpy as np
from scipy.optimize import minimize
from GD_cost_spr import GD_cost_spr

def GD_solver_spr(x0, P, Sm, lambda_, gama, itr_max=500):
    #x0 权重的初始值
    n_m = Sm.shape[1]  # 样本数量
    CostDiff = np.inf #用于记录当前迭代与前一轮迭代损失的差值
    Cost = [10]  # 初始损失值
    # 初始化权重的平方根，并确保是张量
    v = torch.tensor(np.sqrt(x0), dtype=torch.float32, device=Sm.device)  # v 应该是 (40,)

    j = 0  # 迭代计数器

#梯度下降主循环
    while (CostDiff > 0.001 * Cost[j]) and (j < itr_max):
        j += 1
        #当前权重下的预测边界
        p_m = torch.matmul(Sm, v ** 2)  # 确保维度匹配 512,40  40,   p_m:512
        # 逻辑回归的预测值
        Logis = 1 / (1 + torch.exp(-p_m))
        # 逻辑回归的导数（针对负对数损失函数）
        Logis_der = 1 - Logis
        # 损失项梯度
        grad_m = (1 / n_m) * torch.matmul(Sm.T, Logis_der)
        # 完整梯度
        grad = (gama + lambda_ * 2 * torch.matmul(P, v ** 2) - grad_m) * 2 * v
        # 确保 grad 是 PyTorch 张量
        grad = grad.to(torch.float32)

        # 使用线搜索优化步长
        def cost_func(alpha):
            # 将 alpha 转换为 PyTorch 张量
            alpha = torch.tensor(alpha, dtype=torch.float32, device=Sm.device)
            return GD_cost_spr(alpha, P, Sm, v, grad, lambda_, gama)
        # 使用 scipy minimize 函数来最小化损失
        result = minimize(cost_func, 0, bounds=[(0, 1)], options={'disp': False})
        alpha = result.x[0]
        # 记录损失值
        Cost.append(result.fun)

        # 更新权重
        v = v - alpha * grad
        # 计算损失差异
        CostDiff = torch.abs(torch.tensor(Cost[j], dtype=torch.float32) - torch.tensor(Cost[j - 1], dtype=torch.float32))

    # 最终的权重与输出
    w = v ** 2  # 计算最终的权重——通过平方根反推出原始权重
    return w, Cost[-1]
