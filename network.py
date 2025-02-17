import torch.nn as nn
from torch.nn.functional import normalize
import torch
from sklearn.cluster import KMeans

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# SCMVC Network
class Network(nn.Module):
    def __init__(self, view, input_size,num_samples,num_clusters, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.view = view
        self.num_samples = num_samples
        self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()
        self.num_clusters = num_clusters

        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        #全局特征融合层：将不同视图的低级特征（zs）融合为一个全局特征H
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(self.view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        #视图共识特征学习层：从每个视图的特征中提取共识特征r
        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

        #增加一个softmax算r的概率qs
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(high_feature_dim, num_clusters),
            nn.Softmax(dim=1)
        )

    #特征融合和视图共识特征学习
    def feature_fusion(self, zs, zs_gradient):
        input = torch.cat(zs, dim=1) if zs_gradient else torch.cat(zs, dim=1).detach()
        return normalize(self.feature_fusion_module(input),dim=1)

    def forward(self, xs, zs_gradient=True):
        rs = [] #各视图的视图共识特征
        xrs = [] #重构视图
        zs = [] #各视图的低级特征
        qs=[] #每个视图的聚类概率分布

        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_module(z),dim=1) #视图共识特征r
            q = self.label_contrastive_module(r)  # 通过标签对比模块生成聚类概率分布q
            rs.append(r)
            qs.append(q)
            zs.append(z)
            xrs.append(xr)
        #全局特征融合
        H = self.feature_fusion(zs,zs_gradient)
        return xrs,zs,rs,H,qs


    #聚类
    def clustering(self, features):
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=100)
        psedo_labels = kmeans.fit_predict(features.cpu().data.numpy()) #对H进行kmeans之后得到的伪标签
        return psedo_labels