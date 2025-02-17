import torch

def loss_function(f, Sm, P, lambda_, gamma_):
    """
    计算损失函数：
    min_f sum_i=1^(2M) log(1 + exp(-f^T r(i))) + lambda f^T P f + gamma |f|_1

    f: 联合特征权重向量，包含两个视图的权重
    Sm: 差异矩阵（由Z_x和Z_y构成）——论文中的R
    P: 论文中的D
    lambda_: L2 正则化的超参数
    gamma_: L1 正则化的超参数
    """
    if f.dim() == 1:
        mf = f.unsqueeze(1)

    # 最大化样本边际
    log_loss = torch.sum(torch.log(1 + torch.exp(-torch.matmul(Sm, f))))
    # 最小化跨视图匹配误差
    l2_loss = torch.sum(torch.matmul(mf.mT, torch.matmul(P, f)))
    # 计算L1正则化项 鼓励稀疏性
    l1_loss = torch.sum(torch.abs(f))
    # 计算总损失
    loss = log_loss + lambda_ * l2_loss + gamma_ * l1_loss

    # 在计算损失后进行检查
    if torch.isnan(loss).any():
        print(f"Loss is NaN, f: {f}, Sm: {Sm}, P: {P}, lambda_: {lambda_}, gamma_: {gamma_}")
        print(f"log_loss: {log_loss}, l2_loss: {l2_loss}, l1_loss: {l1_loss}")


    return loss.item()