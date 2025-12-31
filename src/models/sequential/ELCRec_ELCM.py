import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from models.BaseModel import SequentialModel
from utils import layers

# ==========================================
# 1. 复制 ELCRecBase 基础逻辑，确保独立运行
# ==========================================
class ELCRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1, help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads.')
        parser.add_argument('--cluster_k', type=int, default=256)
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--w_cl', type=float, default=0.1)
        parser.add_argument('--tau', type=float, default=0.2)
        parser.add_argument('--aug_ratio', type=float, default=0.2)
        parser.add_argument('--fusion', type=str, default='shift', choices=['shift', 'concat'])
        return parser        

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.cluster_k = args.cluster_k
        self.alpha = args.alpha
        self.cluster_dim = self.max_his * self.emb_size
        self.w_cl = args.w_cl
        self.tau = args.tau
        self.aug_ratio = args.aug_ratio
        self.fusion = args.fusion

        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        mask = torch.tril(torch.ones(self.max_his, self.max_his, device=self.device))
        self.register_buffer("causal_mask_full", mask.view(1, 1, self.max_his, self.max_his))
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])
        self.cluster_centers = nn.Parameter(torch.randn(self.cluster_k, self.cluster_dim))
        if getattr(self, "fusion", "shift") == "concat":
            self.fusion_proj = nn.Linear(self.cluster_dim * 2, self.cluster_dim)

    def encode_seq(self, history: torch.Tensor, lengths: torch.Tensor):
        batch_size, seq_len = history.shape
        lengths = lengths.long()
        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors
        attn_mask = self.causal_mask_full[:, :, :seq_len, :seq_len]
        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()
        h_concat = his_vectors.reshape(batch_size, -1)
        idx = torch.arange(batch_size, device=self.device)
        his_vector = his_vectors[idx, lengths - 1, :]
        return his_vector, h_concat

    def forward(self, feed_dict):
        i_ids = feed_dict['item_id']
        history = feed_dict['history_items']
        lengths = feed_dict['lengths'].long()
        his_vector, h_concat = self.encode_seq(history, lengths)
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        u_v = his_vector.repeat(1, i_ids.shape[1]).view(i_ids.shape[0], i_ids.shape[1], -1)
        out = {
            'prediction': prediction.view(history.size(0), -1),
            'u_v': u_v,
            'i_v': i_vectors,
            'h_concat': h_concat,
        }
        return out

    def cluster_loss(self, h_concat: torch.Tensor) -> torch.Tensor:
        h = F.normalize(h_concat, dim=-1)
        c = F.normalize(self.cluster_centers, dim=-1)
        k = c.size(0)
        diff_cc = c.unsqueeze(1) - c.unsqueeze(0)
        dist2_cc = (diff_cc * diff_cc).sum(dim=-1)
        eye = torch.eye(k, device=dist2_cc.device, dtype=torch.bool)
        decouple = -dist2_cc[~eye].mean()
        diff_hc = h.unsqueeze(1) - c.unsqueeze(0)
        dist2_hc = (diff_hc * diff_hc).sum(dim=-1)
        align = dist2_hc.mean()
        return decouple + align

# ==========================================
# 2. 定义消融实验模型：仅保留聚类 (B + ELCM)
# ==========================================
class ELCRec_ELCM(SequentialModel, ELCRecBase):
    """
    消融实验变体1: B + ELCM
    仅保留端到端可学习聚类模块 (Cluster Loss)，去除意图辅助对比学习 (ICL)。
    """
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'cluster_k', 'alpha']

    @staticmethod
    def parse_model_args(parser):
        parser = ELCRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        # 直接使用 Base 的 forward，不进行任何数据增强计算
        return ELCRecBase.forward(self, feed_dict)

    def loss(self, out_dict, *args, **kwargs):
        # 1. 计算基础推荐损失 (Next Item Prediction)
        loss_next = SequentialModel.loss(self, out_dict, *args, **kwargs)

        # 2. 计算聚类损失 (ELCM)
        loss_cluster = self.cluster_loss(out_dict['h_concat'])

        # 3. 总损失 = 推荐损失 + alpha * 聚类损失
        # 忽略对比学习部分
        loss = loss_next + self.alpha * loss_cluster
        
        return loss