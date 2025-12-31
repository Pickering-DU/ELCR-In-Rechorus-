import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from models.BaseModel import SequentialModel
from utils import layers

# ==========================================
# 1. 复制 ELCRecBase (为了独立运行，必须包含)
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

    def augment(self, seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        # 简单实现 augment，保证 ICL 能跑
        # 这里集成 mask/crop/reorder
        op = random.choice(["mask", "crop", "reorder"])
        B, L = seq.size()
        if op == "mask":
            aug = seq.clone()
            valid = (aug != pad_id)
            rand = torch.rand_like(aug.float())
            mask_pos = (rand < self.aug_ratio) & valid
            aug[mask_pos] = pad_id
            return aug
        elif op == "crop":
            out = seq.new_full((B, L), pad_id)
            for b in range(B):
                valid = seq[b][seq[b] != pad_id]
                n = valid.size(0)
                if n <= 2:
                    out[b, :n] = valid; continue
                keep = max(1, int(n * (1 - self.aug_ratio)))
                start = torch.randint(0, n - keep + 1, (1,), device=seq.device).item()
                out[b, :keep] = valid[start:start+keep]
            return out
        else: # reorder
            out = seq.clone()
            for b in range(B):
                valid_idx = (out[b] != pad_id).nonzero(as_tuple=False).view(-1)
                n = valid_idx.size(0)
                if n <= 3: continue
                seg_len = max(1, int(n * self.aug_ratio))
                start = torch.randint(0, n - seg_len + 1, (1,), device=seq.device).item()
                seg = valid_idx[start:start+seg_len]
                perm = seg[torch.randperm(seg_len, device=seq.device)]
                out[b, seg] = out[b, perm]
            return out

    def forward_with_aug(self, feed_dict):
        # 带有数据增强的 forward
        i_ids = feed_dict['item_id']
        history = feed_dict['history_items']
        lengths = feed_dict['lengths'].long()
        
        his_vector, h_concat = self.encode_seq(history, lengths)
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        
        out = {
            'prediction': prediction.view(history.size(0), -1),
            'h_concat': h_concat,
        }

        # 如果需要计算对比损失，计算增强视图
        if self.training:
            aug1 = self.augment(history)
            aug2 = self.augment(history)
            
            # 简单计算增强后的长度
            valid1 = (aug1 > 0).long()
            pos = torch.arange(1, aug1.size(1) + 1, device=aug1.device).unsqueeze(0)
            len1 = (valid1 * pos).max(dim=1).values.clamp(min=1)
            
            valid2 = (aug2 > 0).long()
            len2 = (valid2 * pos).max(dim=1).values.clamp(min=1)

            z1, h1_concat = self.encode_seq(aug1, len1)
            z2, h2_concat = self.encode_seq(aug2, len2)
            
            out.update({
                'h1_concat': h1_concat,
                'h2_concat': h2_concat
            })
        return out

    def info_nce(self, z1, z2):
        z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
        logits12 = (z1 @ z2.t()) / self.tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits12, labels) + F.cross_entropy(logits12.t(), labels))

    def nearest_center_idx(self, h):
        dist2 = ((h[:, None, :] - self.cluster_centers[None, :, :]) ** 2).sum(-1)
        return dist2.argmin(dim=1)

    def fuse_intent(self, h, c):
        if self.fusion == "shift": return h + c
        else: return self.fusion_proj(torch.cat([h, c], dim=-1))

    def intent_cl_loss(self, h_fused, idx):
        h = F.normalize(h_fused, dim=-1)
        c = F.normalize(self.cluster_centers, dim=-1)
        logits = (h @ c.t()) / self.tau
        return F.cross_entropy(logits, idx)

# ==========================================
# 2. 定义消融实验模型：仅保留对比 (B + ICL)
# ==========================================
class ELCRec_ICL(SequentialModel, ELCRecBase):
    """
    消融实验变体2: B + ICL
    仅保留意图辅助对比学习 (ICL)，去除显式的聚类约束损失 (Cluster Loss)。
    """
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'cluster_k', 'w_cl', 'tau']

    @staticmethod
    def parse_model_args(parser):
        parser = ELCRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        # 使用带有增强逻辑的 forward
        return self.forward_with_aug(feed_dict)

    def loss(self, out_dict, *args, **kwargs):
        # 1. 计算基础推荐损失
        loss_next = SequentialModel.loss(self, out_dict, *args, **kwargs)

        # 2. 计算对比学习损失 (ICL)
        # 只有在训练且权重大于0时才计算
        if self.training and (self.w_cl > 0):
            idx = self.nearest_center_idx(out_dict['h1_concat'])
            c_idx = self.cluster_centers[idx]

            # 融合意图信息
            h1_f = self.fuse_intent(out_dict['h1_concat'], c_idx)
            h2_f = self.fuse_intent(out_dict['h2_concat'], c_idx)

            # 序列对比损失 (SeqCL)
            loss_seqcl = self.info_nce(h1_f, h2_f)

            # 意图对比损失 (IntentCL)
            loss_intent1 = self.intent_cl_loss(h1_f, idx)
            loss_intent2 = self.intent_cl_loss(h2_f, idx)
            loss_intent = 0.5 * (loss_intent1 + loss_intent2)

            # 总损失 = 推荐损失 + w_cl * (序列对比 + 意图对比)
            # 关键：这里去掉了 cluster_loss
            loss = loss_next + self.w_cl * (loss_seqcl + loss_intent)
        else:
            loss = loss_next

        return loss