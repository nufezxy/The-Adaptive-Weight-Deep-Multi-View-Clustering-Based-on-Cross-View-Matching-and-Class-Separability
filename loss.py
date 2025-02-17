import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size #每个批次中的样本数，决定了正样本和负样本的计算范围
        self.temperature = temperature #温度参数，用于调节对比损失的分布敏感性。较低的温度会放大相似性差异。
        self.device = device

    def forward(self, h_i, h_j, weight=None): #h_i，h_j：来自不同视图的嵌入向量； weight：可选的加权参数，用于对最终损失进行加权
        N =self.batch_size
        #计算两个特征集合相似性
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        #提取正样本对相似度
        positives = torch.diag(similarity_matrix)
        #创建负样本掩码
        mask = torch.ones((N, N)).to(self.device) #创建一个全为1的矩阵
        mask = mask.fill_diagonal_(0) #对角线设置为0
        nominator = torch.exp(positives) #分子，正样本
        denominator = (mask.bool()) * torch.exp(similarity_matrix) #分母，负样本对的指数化相似度（仅保留掩码中为 1 的部分）
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / N
        #加权处理
        loss = weight * loss if weight is not None else loss

        return loss




