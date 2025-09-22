import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import networkx as nx
import math


def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class CSFM(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        return fuse

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class PSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

class WSGCN(nn.Module):
    def __init__(
            self,
            in_channels=64,
            feat_dim=256 * 256,
            hidden_dim=256,
            k_neighbors=4,
            rewire_prob=0.3,
            heads=8,
            concat=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_down = nn.Linear(feat_dim, hidden_dim)
        self.gat = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=concat)
        self.linear_up = nn.Linear(hidden_dim, feat_dim)
        self.edge_index = self.generate_ws_edges(
            num_nodes=in_channels,
            k=k_neighbors,
            p=rewire_prob
        )

    def generate_ws_edges(self, num_nodes, k=4, p=0.3):
        ws_graph = nx.connected_watts_strogatz_graph(
            n=num_nodes, k=k, p=p, tries=100
        )
        edge_index = torch.tensor(
            list(ws_graph.edges()), dtype=torch.long
        ).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        x_down = self.linear_down(x_flat)
        x_down = x_down.view(-1, self.hidden_dim)
        x_gat = self.gat(x_down, self.edge_index.to(x.device))
        x_gat = x_gat.view(B, C, -1)
        x_up = self.linear_up(x_gat)
        x_out = x_up.view(B, C, H, W)
        return x_out

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.psconv1 = PSConv(1, 16, k=3, s=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.psconv2 = PSConv(16, 32, k=3, s=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.psconv3 = PSConv(32, 64, k=3, s=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.psconv4 = PSConv(64, 128, k=3, s=1)
        self.GNN=WSGCN(in_channels=128, feat_dim=32 * 32,hidden_dim=32,k_neighbors=4, rewire_prob=0.3)
        self.concat1=CSFM(in_channel=16)
        self.concat2=CSFM(in_channel=32)
        self.concat3=CSFM(in_channel=64)
        self.concat4=CSFM(in_channel=128)
    def forward(self,x_conv,down_x,x_mri):
        down_x1,down_x2,down_x3=down_x
        fused_features = []
        x0 = self.psconv1(x_mri)
        x0_concat=self.concat1(x0,x_conv)
        conv_final=x0_concat
        x0 = self.maxpool1(x0)
        x1 = self.psconv2(x0)
        X1_concat=self.concat2(x1,down_x1)
        X1_final=X1_concat
        x2 = self.maxpool2(x1)
        x2 = self.psconv3(x2)
        X2_concat=self.concat3(x2,down_x2)
        X2_final=X2_concat
        x3 = self.maxpool3(x2)
        x3 = self.psconv4(x3)
        X3_concat=self.concat4(x3,down_x3)
        X3_final=X3_concat
        x3_add=x3+down_x3
        x3_add=self.GNN(x3_add)

        return conv_final,[X1_final, X2_final,X3_final],x3_add