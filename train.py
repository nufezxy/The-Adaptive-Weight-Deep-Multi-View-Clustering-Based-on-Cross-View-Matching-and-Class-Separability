import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from Newloss import loss_function
from MvSV import MvSV  # 导入 MvSV 方法
from dataloader import load_data

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# Cifar10
# Cifar100
# Prokaryotic
# Synthetic3d
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0001)#0.0001
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=10)  #记得改回来200
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--temperature", default=1)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 10 # 20
    seed = 30
if args.dataset == "CCV":
    args.con_epochs = 50 # 100
    seed = 100
    args.tune_epochs = 200
if args.dataset == "Fashion":
    args.con_epochs = 50 # 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 200
    args.tune_epochs = 200
if args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 30
if args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 100
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 1000000
if args.dataset == "Cifar10":
    args.con_epochs = 10
    seed = 10
if args.dataset == "Cifar100":
    args.con_epochs = 20
    seed = 10
if args.dataset == "Prokaryotic":
    args.con_epochs = 20
    seed = 10000
if args.dataset == "Synthetic3d":
    args.con_epochs = 100
    seed = 100

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed=1
setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

#根据模型的聚类方法，生成伪标签
def psedo_labeling(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    commonH_list = []
    for batch_idx, (xs, y, _) in enumerate(loader):
        xs = [x.to(device) for x in xs]
        with torch.no_grad():
            xrs, zs, rs, H,qs= model(xs) #输入多视图数据
            commonH_list.append(H)
    commonH = torch.cat(commonH_list, dim=0)
    #对公共表示 commonZ 执行聚类操作，生成伪标签
    psedo_labels = model.clustering(commonH)
    #存储在psedo_labels中
    model.psedo_labels = psedo_labels

#预训练
def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs,_,_,_ ,_= model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

#对比学习训练
def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    #新增
    cross_entropy = torch.nn.CrossEntropyLoss()
    psedo_labeling(model, dataset, args.batch_size)
    for batch_idx, (xs, _, sample_idx) in enumerate(data_loader):
        # 从伪标签中提取当前批次样本的标签
        y_pred = model.psedo_labels[sample_idx]
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, H,qs = model(xs)  #H：256，20   R：256，20
        loss_list = []
        for v in range(view):
            # 调用 MvSV 计算 Sm 和 P
            f,Sm, P,fr,fh = MvSV(rs[v], H, y_pred, 2, 0.0001, 0.01)  # 传递 X, Y, id 和超参数0.0001  0.01
            # 扩展 fr，使其与 rs[v] 具有相同的形状
            fr_expanded = fr.expand_as(rs[v])  # 将 fr 扩展成与 rs[v] 相同的形状 (256, 20)
            # 使用 fr 来代替 w 计算 contrastive loss
            loss_list.append(torch.mean(contrastiveloss(H, rs[v], fr_expanded)))  # 求均值mean
            # 重建损失
            loss_list.append(mse(xs[v], xrs[v]))

            # 将 qs[v] 和 y_pred 都移到相同的设备上
            qs_v_tensor = qs[v].to(device)
            y_pred_tensor = torch.from_numpy(y_pred).long().to(device)
            # 计算交叉熵损失
            loss_list.append(cross_entropy(qs_v_tensor, y_pred_tensor))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))



accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1

#模型训练循环
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims,data_size,class_num, args.feature_dim, args.high_feature_dim, device)
    print(model)
    model = model.to(device)
    state = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)
    best_acc, best_nmi, best_pur = 0, 0, 0

    epoch = 1
    while epoch <= args.pre_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.pre_epochs + args.con_epochs:
        contrastive_train(epoch)
        acc, nmi, pur = valid(model, dataset, data_size)
        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
        epoch += 1

    # The final result
    accs.append(best_acc)
    nmis.append(best_nmi)
    purs.append(best_pur)
    print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))
