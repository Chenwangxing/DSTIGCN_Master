import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable




class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False):
        super(DeformConv3d, self).__init__()
        N = kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.kernel_size0 = kernel_size[0]
        self.kernel_size1 = kernel_size[1]
        self.kernel_size2 = kernel_size[2]
        self.padding0 = padding[0]
        self.padding1 = padding[1]
        self.padding2 = padding[2]
        self.stride = stride
        self.zero_padding = nn.ZeroPad3d((padding[1], padding[1], padding[2], padding[2], padding[0], padding[0]))

        self.conv_kernel = nn.Conv3d(in_channels * N, out_channels, kernel_size=1, bias=bias)
        self.offset_conv_kernel = nn.Conv3d(in_channels, N * 3, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        # x [1, 4-num_head, 8-T, N, N]
        offset = self.offset_conv_kernel(x)
        # offset [1, 3 * kernel_size[0] * kernel_size[1] * kernel_size[2], 8-T, N, N]

        dtype = offset.data.type()
        ks0 = self.kernel_size0
        ks1 = self.kernel_size1
        ks2 = self.kernel_size2
        N = offset.size(1) // 3

        if self.padding0:
            x = self.zero_padding(x)   # x [1, 4-num_head, T+padding, N, N]

        p = self._get_p(offset, dtype)
        p = p[:, :, ::self.stride, ::self.stride, ::self.stride]  # p [1, 9, T, N, N]

        # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
        p = p.contiguous().permute(0, 2, 3, 4, 1)  # 5D array

        # 8 neighbor points with integer coords
        # (b, h, w, d, N)
        mask = torch.cat([
            p[..., :N].lt(self.padding0) + p[..., :N].gt(x.size(2) - 1 - self.padding0),
            p[..., N:2 * N].lt(self.padding1) + p[..., N:2 * N].gt(x.size(3) - 1 - self.padding1),
            p[..., 2 * N:].lt(self.padding2) + p[..., 2 * N:].gt(x.size(4) - 1 - self.padding2),
        ], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))  # все еще непонятно, что тут происходит за wtf
        p = p * (1 - mask) + floor_p * mask

        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1),
        ], dim=-1)

        # kernel (b, h, w, d, N)
        p = torch.round(p).long()
        x_offset = self._get_x_q(x, p, N)  # x_offset [1, 4, 8-T, N, N, 3]
        x_offset = self._reshape_x_offset(x_offset, ks0, ks1, ks2)  # x_offset3 [1, 12(4*3), T, N, N]
        out = self.conv_kernel(x_offset)
        return out
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size0 - 1) // 2, (self.kernel_size0 - 1) // 2 + 1),
            range(-(self.kernel_size1 - 1) // 2, (self.kernel_size1 - 1) // 2 + 1),
            range(-(self.kernel_size2 - 1) // 2, (self.kernel_size2 - 1) // 2 + 1),
            indexing='ij')
        # (3N, 1) - 3 coords for each of N offsets
        # (x1, ... xN, y1, ... yN, z1, ... zN)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        # p_n (1, 9, 1, 1, 1)
        p_n = torch.from_numpy(p_n).type(dtype)
        return p_n
    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)   # p_0_x (1, 3, 8, 3, 3)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)   # p_0_y (1, 3, 8, 3, 3)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)   # p_0_z (1, 3, 8, 3, 3)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)   # p_0_z (1, 9, 8, 3, 3)
        p_0 = torch.from_numpy(p_0).type(dtype)
        return p_0
    def _get_p(self, offset, dtype):
        # offset [1, 3 * kernel_size[0] * kernel_size[1] * kernel_size[2], T, N, N]
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)
        # N=3 * kernel_size[0] * kernel_size[1] * kernel_size[2];   h=T;   w=N;   d=N
        p_n = self._get_p_n(N, dtype).to(offset)  #  p_n (1, 3*kernel1*kernel2*kernel3, 1, 1, 1)
        # p_0 (1, 3*kernel1*kernel2*kernel3, T, N, N)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset
        return p
    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        # x.size == (b, c, h, w, d)
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        # (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, d, N)
        # offset_x * w * d + offset_y * d + offset_z
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)
        return x_offset
    @staticmethod
    def _reshape_x_offset(x_offset, ks0, ks1, ks2):
        # x_offset [1, 4, 8-T, N, N, 3]
        b, c, h, w, d, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        # x_offset [1, 4, 3, 8-T, N, N]
        x_offset = x_offset.contiguous().view(b, c * N, h, w, d)
        return x_offset


##### Deformable spatial-temporal interaction convolution #####
class AsymmetricConvolution(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()
        self.convd3d1 = DeformConv3d(in_cha, out_cha, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)
        self.convd3d2 = DeformConv3d(in_cha, out_cha, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)
        self.convd3d3 = DeformConv3d(in_cha, out_cha, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)
        self.shortcut = lambda x: x
        if in_cha != out_cha:
            self.shortcut = nn.Conv3d(in_cha, out_cha, 1, bias=False)

        self.activation = nn.PReLU()
    def forward(self, x):
        # ([8obs, 4heads, N, N])
        x = x.permute(1, 0, 2, 3)
        x = x.unsqueeze(0)   # x [1, 4-heads, 8-T, N, N]
        shortcut = self.shortcut(x)

        #### Deformable spatial-temporal interaction convolution ####
        xd3d1 = self.convd3d1(x)   # xd3d1 [1, 4-heads, 8-T, N, N]
        xd3d2 = self.convd3d2(x.permute(0, 1, 3, 2, 4))   # xd3d2 [1, 4-heads, N, 8-T, N]
        xd3d3 = self.convd3d3(x.permute(0, 1, 4, 3, 2))   # xd3d3 [1, 4-heads, N, N, 8-T]

        xd3d = self.activation(xd3d1 + xd3d2.permute(0, 1, 3, 2, 4) + xd3d3.permute(0, 1, 4, 3, 2))
        x2 = xd3d + shortcut
        x2 = x2.squeeze(0)
        return x2.permute(1, 0, 2, 3)


class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]

    def forward(self, x, mask=False, multi_head=False):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model
        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)
        attention = self.softmax(attention / self.scaled_factor)
        if mask is True:
            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)
        return attention


class SparseWeightedAdjacency(nn.Module):
    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, obs_len=8, dropout=0):
        super(SparseWeightedAdjacency, self).__init__()
        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.st_convolutions = nn.ModuleList()
        for i in range(3):
            self.st_convolutions.append(AsymmetricConvolution(4, 4))

        self.dropout = dropout
        self.st_output = nn.Sigmoid()
        self.spa_softmax = nn.Softmax(dim=-1)
    def forward(self, graph, identity):
        assert len(graph.shape) == 3
        spatial_graph = graph[:, :, 1:]  # (T N 2)
        # (T num_heads N N)
        dense_spatial_interaction = self.spatial_attention(spatial_graph, multi_head=True)
        # dense_spatial_interaction  torch.Size([8obs, 4heads, N, N])

        ##### Deformable spatial-temporal interaction module #####
        for j in range(3):
            dense_spatial_interaction = self.st_convolutions[j](dense_spatial_interaction)
        st_interaction_mask = self.st_output(dense_spatial_interaction)

        st_mask = st_interaction_mask + identity[0].unsqueeze(1)
        normalized_spatial_adjacency_matrix = self.spa_softmax(dense_spatial_interaction * st_mask)
        return normalized_spatial_adjacency_matrix




class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()
        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout
    def forward(self, graph, adjacency):
        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)
        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()
        self.dropout = dropout
        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))

    def forward(self, graph, normalized_spatial_adjacency_matrix):
        # graph [1 seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        return gcn_spatial_features.permute(2, 0, 1, 3)


class GateTCN(nn.Module):
    def __init__(self, in_dims=8, out_dims=12):
        super(GateTCN, self).__init__()
        self.tcn = nn.Conv2d(in_dims, out_dims, 3, padding=1)
        self.activation1 = nn.PReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(out_dims, out_dims, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, features):
        tcn_features = self.activation1(self.tcn(features))
        gate = self.avg_pool(features) + self.max_pool(features)
        gate = self.linear(gate)
        return tcn_features * gate


class TrajectoryModel(nn.Module):
    def __init__(self,number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5, out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()
        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout
        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency()
        # graph convolution
        self.stsgcn = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)

        self.tcns = nn.ModuleList()
        self.tcns.append(GateTCN(8, 12))
        for j in range(1, 5):
            self.tcns.append(GateTCN(12, 12))

        self.output = nn.Linear(embedding_dims // num_heads, out_dims)

    def forward(self, graph, identity):
        # graph 1 obs_len N 3
        # Sparse Graph Learning部分         temporal_embeddings(N行人数目 T时间8 d_model64)dense_temporal_interaction
        normalized_st_adjacency_matrix = self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)

        gcn_st_features = self.stsgcn(graph, normalized_st_adjacency_matrix)   # gcn_st_features [2-N, 8-Tobs, 4-heads, 16-feat]

        features = self.tcns[0](gcn_st_features)
        for i in range(1, 5):
            features = F.dropout(self.tcns[i](features), p=0.1) + features
        #  features [2-N, 12-Tprd, 40head, 5-feat]

        features = torch.mean(self.output(features), dim=-2)

        return features.permute(1, 0, 2).contiguous()
