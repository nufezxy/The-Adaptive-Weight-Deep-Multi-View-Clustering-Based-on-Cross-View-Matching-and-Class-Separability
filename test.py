import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data


Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--hide_feature_dim", default=20)
parser.add_argument("--high_feature_dim", default=20, type=int, help="High feature dimension")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims,data_size,class_num, args.feature_dim, args.high_feature_dim, device)
model = model.to(device)

checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)

model.eval()
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
acc, nmi, pur= valid(model, dataset, data_size)
