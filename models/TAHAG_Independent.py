import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from utils.utils import mmd

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class addResidual(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size = 1, residual = True):
        super(addResidual, self).__init__()

        if residual:
            if in_feats != out_feats:
                self.res = nn.Sequential(
                    nn.Conv1d(in_feats, out_feats, kernel_size=kernel_size),
                    nn.BatchNorm1d(out_feats)
                )
            else:
                self.res = lambda x: x
        else:
            self.res = lambda x: 0

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)

    def forward(self, x):
        out = self.res(x)
        return out

class unit_gcn(nn.Module):
    def __init__(self, in_feats, out_feats, A, coff_emb = 4, adaptive = True, attention = True, residual = False):
        super(unit_gcn, self).__init__()
        inter_feats = out_feats // coff_emb
        self.inter_f = inter_feats
        self.out_f = out_feats
        self.in_f = in_feats

        self.conv_d = nn.Conv1d(in_feats, out_feats, kernel_size=1)

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

            self.conv_a = nn.Conv1d(in_feats, inter_feats, kernel_size=1)
            self.conv_b = nn.Conv1d(in_feats, inter_feats, kernel_size=1)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad = False)
        self.adaptive = adaptive

        if attention:
            reduction = 2
            self.fc1_fa = nn.Linear(out_feats, out_feats//reduction)
            self.fc2_fa = nn.Linear(out_feats//reduction, out_feats)
            nn.init.kaiming_normal_(self.fc1_fa.weight)
            nn.init.constant_(self.fc1_fa.bias, 0)
            nn.init.constant_(self.fc2_fa.weight, 0)
            nn.init.constant_(self.fc2_fa.bias, 0)

        self.attention = attention

        self.add_residual_gcn = addResidual(in_feats, out_feats, kernel_size=1, residual=True)
        self.add_residual_global = addResidual(in_feats, out_feats, kernel_size=1, residual=residual)
        self.residual = residual

        self.bn = nn.BatchNorm1d(out_feats)
        self.global_bn = nn.BatchNorm1d(out_feats)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.global_relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        batch, n_feats, n_nodes = x.shape

        if self.adaptive:
            A = self.PA
            A1 = self.conv_a(x).permute(0, 2, 1)
            A2 = self.conv_b(x)
            A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))
            A3 = A + A1 * self.alpha
            A2 = x
            y = self.conv_d(torch.matmul(A2, A3))
        else:
            A = self.A.cuda(x.get_device())
            A2 = x
            A3 = A
            y = self.conv_d(torch.matmul(A2, A3))

        y = self.bn(y)
        y += self.add_residual_gcn(x)
        y = self.relu(y)

        if self.attention:
            fe = y.mean(-1)
            fe1 = self.relu(self.fc1_fa(fe))
            fe2 = self.sigmoid(self.fc2_fa(fe1))
            y = y * fe2.unsqueeze(-1) + y

        y = self.global_bn(y)
        if self.residual:
            y += self.add_residual_gcn(x)
        y = self.global_relu(y)

        return y

class sharedNet(nn.Module):
    def __init__(self, in_feats, A, adaptive, attention):
        super(sharedNet, self).__init__()
        self.data_bn = nn.BatchNorm1d(in_feats)
        bn_init(self.data_bn, 1)
        self.l1 = unit_gcn(in_feats, 32, A, adaptive=adaptive, attention=attention, residual=True)
        self.l2 = unit_gcn(32, 64, A, adaptive=adaptive, attention=attention, residual=True)
        self.l3 = unit_gcn(64, 128, A, adaptive=adaptive, attention=attention, residual=True)

    def forward(self, x):
        x = torch.reshape(x, (-1, 62, 5))
        x = x.transpose(2, 1)
        x = self.data_bn(x)
        x = self.l1(x)
        hid_feats1 = torch.reshape(x, (-1, 62*32))
        x = self.l2(x)
        hid_feats2 = torch.reshape(x, (-1, 62*64))
        x = self.l3(x)
        x = x.mean(-1)

        return x, hid_feats1, hid_feats2

class Model(nn.Module):
    def __init__(self, in_feats, num_class = 3, adj = None, drop_out = 0, adaptive = True, attention = True):
        super(Model, self).__init__()
        A = adj
        self.num_class = num_class
        self.sharedNet = sharedNet(in_feats, A, adaptive, attention)
        self.domain_classifier = nn.Linear(128, 2)
        self.fc = nn.Linear(128, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x_src, x_tgt, alpha = 0.01):
        mmd_loss = 0.
        domain_out = None
        src_feats, src_hid_feat1, src_hid_feat2 = self.sharedNet(x_src)

        if self.training == True:
            tgt_feats, tgt_hid_feat1, tgt_hid_feat2 = self.sharedNet(x_tgt)
            mmd_loss1 = mmd(src_hid_feat1, tgt_hid_feat1)
            mmd_loss2 = mmd(src_hid_feat2, tgt_hid_feat2)
            mmd_loss = mmd_loss + mmd_loss1 + mmd_loss2

            reverse_src = ReverseLayerF.apply(src_feats, alpha)
            domain_src_out = self.domain_classifier(reverse_src)
            reverse_tgt = ReverseLayerF.apply(tgt_feats, alpha)
            domain_tgt_out = self.domain_classifier(reverse_tgt)
            domain_out = torch.cat([domain_src_out, domain_tgt_out])

        # classifier.
        src_feats2 = self.drop_out(src_feats)
        cls_out = self.fc(src_feats2)

        return cls_out, domain_out, mmd_loss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn([10, 62, 5]).to(device)
    adj = np.eye(62)
    model = Model(5, 3, adj, 0.5, True, True).to(device)
    out = model(x)
    print(out.shape)
