import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers

class ELCRecBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.')
		parser.add_argument('--num_heads', type=int, default=2,
							help='Number of attention heads.')
		parser.add_argument('--cluster_k', type=int, default=256)
		parser.add_argument('--alpha', type=float, default=0.1)
		#parser.add_argument('--tau', type=float, default=0.2)
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
		self.cluster_dim = self.max_his * self.emb_size   # D = L*d
		self.w_cl = args.w_cl
		self.tau = args.tau
		self.aug_ratio = args.aug_ratio
		self.fusion = args.fusion


		self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
		# 缓存最大的因果mask
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
		"""
		history: [B, L]  (padding=0)
		lengths: [B]
		return:
		  his_vector: [B, d]     # 用最后一个有效位置的表示
		  h_concat:   [B, L*d]   # concat pooling，用于聚类/intent
		"""
		batch_size, seq_len = history.shape
		lengths = lengths.long()

		valid_his = (history > 0).long()                          # [B, L]
		his_vectors = self.i_embeddings(history)                  # [B, L, d]

		# position embedding
		position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
		pos_vectors = self.p_embeddings(position)
		his_vectors = his_vectors + pos_vectors

		# self-attention
		#causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
		#attn_mask = torch.from_numpy(causality_mask).to(self.device)
		attn_mask = self.causal_mask_full[:, :, :seq_len, :seq_len]
		for block in self.transformer_block:
			his_vectors = block(his_vectors, attn_mask)

		his_vectors = his_vectors * valid_his[:, :, None].float() # [B, L, d]

		h_concat = his_vectors.reshape(batch_size, -1)            # [B, L*d]

		idx = torch.arange(batch_size, device=self.device)
		his_vector = his_vectors[idx, lengths - 1, :]             # [B, d]
		return his_vector, h_concat


	def forward(self, feed_dict):
		i_ids = feed_dict['item_id']                 # [B, neg+1]
		history = feed_dict['history_items']         # [B, L]
		lengths = feed_dict['lengths'].long()        # [B]

		# (A) 原序列：用于推荐 + 聚类
		his_vector, h_concat = self.encode_seq(history, lengths)

		# (B) 两个增强视图：用于对比学习（seqCL）
		#aug1 = self.augment(history)            # [B, L]
		#aug2 = self.augment(history)            # [B, L]
		#z1, _ = self.encode_seq(aug1, lengths)       # [B, d]
		#z2, _ = self.encode_seq(aug2, lengths)       # [B, d]

		# (C) 推荐打分
		i_vectors = self.i_embeddings(i_ids)         # [B, neg+1, d]
		prediction = (his_vector[:, None, :] * i_vectors).sum(-1)

		u_v = his_vector.repeat(1, i_ids.shape[1]).view(i_ids.shape[0], i_ids.shape[1], -1)
		i_v = i_vectors

		out = {
			'prediction': prediction.view(history.size(0), -1),
			'u_v': u_v,
			'i_v': i_v,
			'h_concat': h_concat,
		}

		# 只有训练阶段才算对比学习
		if (not self.training) or (self.w_cl <= 0):
			return out

		# (B) 两个增强视图：用于对比学习（seqCL / intentCL）
		aug1 = self.augment(history)
		aug2 = self.augment(history)

		B, L = aug1.size()
		pos = torch.arange(1, L + 1, device=aug1.device).unsqueeze(0).expand(B, -1)

		valid1 = (aug1 > 0).long()
		valid2 = (aug2 > 0).long()

		len1 = (valid1 * pos).max(dim=1).values.clamp(min=1)
		len2 = (valid2 * pos).max(dim=1).values.clamp(min=1)

		z1, h1_concat = self.encode_seq(aug1, len1)
		z2, h2_concat = self.encode_seq(aug2, len2)

		# intent 分支会在第 5 步加（先把需要的东西返回）
		out.update({
			'z1': z1,
			'z2': z2,
			'h1_concat': h1_concat,
			'h2_concat': h2_concat,
		})

		return out

	


	def cluster_loss(self, h_concat: torch.Tensor) -> torch.Tensor:
		# h_concat: [B, D], centers: [K, D]
		h = F.normalize(h_concat, dim=-1)
		c = F.normalize(self.cluster_centers, dim=-1)

		k = c.size(0)

		# decouple: 推开簇中心（中心之间越远越好，所以取负号）
		diff_cc = c.unsqueeze(1) - c.unsqueeze(0)      # [K, K, D]
		dist2_cc = (diff_cc * diff_cc).sum(dim=-1)     # [K, K]
		eye = torch.eye(k, device=dist2_cc.device, dtype=torch.bool)
		decouple = -dist2_cc[~eye].mean()

		# align: 拉近行为与中心（对所有中心求平均，避免只拉最近中心）
		diff_hc = h.unsqueeze(1) - c.unsqueeze(0)      # [B, K, D]
		dist2_hc = (diff_hc * diff_hc).sum(dim=-1)     # [B, K]
		align = dist2_hc.mean()

		return decouple + align

	def augment_mask(self, seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
		# seq: [B, L], padding=0
		B, L = seq.size()
		aug = seq.clone()

		# 只对非 padding 位置做 mask
		valid = (aug != pad_id)
		rand = torch.rand_like(aug.float())
		mask_pos = (rand < self.aug_ratio) & valid

		aug[mask_pos] = pad_id   # 用 0 作为 mask（简单、能跑通）
		return aug
	

	def augment_crop(self, seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
		# 对每个样本裁剪一段连续子序列，其余用pad补齐（简单版本）
		B, L = seq.size()
		out = seq.new_full((B, L), pad_id)
		for b in range(B):
			valid = seq[b][seq[b] != pad_id]
			n = valid.size(0)
			if n <= 2:
				out[b, :n] = valid
				continue
			keep = max(1, int(n * (1 - self.aug_ratio)))
			start = torch.randint(0, n - keep + 1, (1,), device=seq.device).item()
			subseq = valid[start:start+keep]
			out[b, :keep] = subseq  # 这里用"右侧pad"的方式，与你当前实现一致
		return out


	def augment_reorder(self, seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
		# 随机选一小段打乱顺序
		B, L = seq.size()
		out = seq.clone()
		for b in range(B):
			valid_idx = (out[b] != pad_id).nonzero(as_tuple=False).view(-1)
			n = valid_idx.size(0)
			if n <= 3:
				continue
			seg_len = max(1, int(n * self.aug_ratio))
			start = torch.randint(0, n - seg_len + 1, (1,), device=seq.device).item()
			seg = valid_idx[start:start+seg_len]
			perm = seg[torch.randperm(seg_len, device=seq.device)]
			out[b, seg] = out[b, perm]
		return out
	

	def augment(self, seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
		op = random.choice(["mask", "crop", "reorder"])
		if op == "mask":
			return self.augment_mask(seq, pad_id)
		if op == "crop":
			return self.augment_crop(seq, pad_id)
		return self.augment_reorder(seq, pad_id)


	
	def info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
		# z1,z2: [B, d]
		z1 = F.normalize(z1, dim=-1)
		z2 = F.normalize(z2, dim=-1)

		logits12 = (z1 @ z2.t()) / self.tau      # [B,B]
		labels = torch.arange(z1.size(0), device=z1.device)

		loss12 = F.cross_entropy(logits12, labels)      # z1 -> z2
		loss21 = F.cross_entropy(logits12.t(), labels)  # z2 -> z1
		return 0.5 * (loss12 + loss21)
	
	def nearest_center_idx(self, h: torch.Tensor) -> torch.Tensor:
		# h: [B, D], centers: [K, D]
		centers = self.cluster_centers
		dist2 = ((h[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # [B,K]
		return dist2.argmin(dim=1)  # [B]

	def fuse_intent(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		# h,c: [B,D]
		if self.fusion == "shift":
			return h + c
		else:
			x = torch.cat([h, c], dim=-1)  # [B,2D]
			return self.fusion_proj(x)     # [B,D]

	def intent_cl_loss(self, h_fused: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
		# 把 centers 当作类别：预测最近中心
		h = F.normalize(h_fused, dim=-1)
		c = F.normalize(self.cluster_centers, dim=-1)
		logits = (h @ c.t()) / self.tau     # [B,K]
		return F.cross_entropy(logits, idx)




class ELCRec(SequentialModel, ELCRecBase):
	reader = 'SeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'num_layers', 'num_heads','cluster_k', 'alpha']

	@staticmethod
	def parse_model_args(parser):
		parser = ELCRecBase.parse_model_args(parser)
		return SequentialModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		SequentialModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = ELCRecBase.forward(self, feed_dict)
		return out_dict

	def loss(self, out_dict, *args, **kwargs):
		# 先拿到原本的 next-item loss（SASRec）
		loss_next = super().loss(out_dict, *args, **kwargs)

		# 计算 cluster loss
		loss_cluster = self.cluster_loss(out_dict['h_concat'])
		loss = loss_next + self.alpha * loss_cluster
		if self.training and (self.w_cl > 0):
			# intent 对比：用 v1 的 h1_concat 找最近中心
			idx = self.nearest_center_idx(out_dict['h1_concat'])
			c_idx = self.cluster_centers[idx]  # [B,D]

			h1_f = self.fuse_intent(out_dict['h1_concat'], c_idx)
			h2_f = self.fuse_intent(out_dict['h2_concat'], c_idx)

			# seqCL：用融合后的向量做（intent-assisted）
			loss_seqcl = self.info_nce(h1_f, h2_f)

			loss_intent1 = self.intent_cl_loss(h1_f, idx)
			loss_intent2 = self.intent_cl_loss(h2_f, idx)
			loss_intent = 0.5 * (loss_intent1 + loss_intent2)

			loss = loss + self.w_cl * (loss_seqcl + loss_intent)
		return loss

class ELCRecImpression(ImpressionSeqModel, ELCRecBase):
	reader = 'ImpressionSeqReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'num_layers', 'num_heads']

	@staticmethod
	def parse_model_args(parser):
		parser = ELCRecBase.parse_model_args(parser)
		return ImpressionSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ImpressionSeqModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return ELCRecBase.forward(self, feed_dict)