import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import MetaTensor
from utils.utils_new import info_if_main


class MemoryBank(nn.Module):
    """基于置信度和存储时间的动态记忆库"""

    def __init__(self, capacity, feat_dim, replace_strategy):
        super().__init__()
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.register_buffer("memory", torch.zeros(capacity, feat_dim))
        self.register_buffer("confidence", torch.zeros(capacity))  # 置信度
        self.register_buffer("epoch", torch.zeros(capacity))  # 存储时间（epoch）
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.replace_strategy = replace_strategy.lower()

    def enqueue(self, features, confidence, current_epoch):
        """将特征和置信度存入记忆库，动态管理"""
        if features.size(0) == 0:
            return
        if isinstance(features, MetaTensor):
            features = features.as_tensor()
        if isinstance(confidence, MetaTensor):
            confidence = confidence.as_tensor()
        features = features.detach()  # 🔹 ​**确保不带梯度**
        confidence = confidence.detach()  # 🔹 ​**确保不带梯度**
        batch_size = features.size(0)
        ptr = self.ptr.item()
        features = features.to(self.memory.device)
        confidence = confidence.to(self.memory.device)

        # 如果记忆库未满，直接存入
        if ptr + batch_size <= self.capacity:
            self.memory[ptr : ptr + batch_size] = features
            self.confidence[ptr : ptr + batch_size] = confidence
            self.epoch[ptr : ptr + batch_size] = current_epoch  # 记录当前 epoch
            self.ptr[0] = ptr + batch_size
        else:
            if self.replace_strategy == "fifo":
                # 先进先出
                # 替换最旧的前 batch_size 个（顺序循环覆盖）
                replace_indices = torch.arange(ptr, ptr + batch_size) % self.capacity
            elif self.replace_strategy == "confidence_only":
                # 替换置信度最低的
                _, replace_indices = torch.topk(self.confidence, batch_size, largest=False)
            elif self.replace_strategy == "cats":
                # 默认策略：置信度 × 时间衰减
                time_diff = current_epoch - self.epoch
                time_diff = torch.clamp(time_diff, min=1)
                weights = self.confidence * (1.0 / time_diff)
                _, replace_indices = torch.topk(weights, batch_size, largest=False)
            else:
                raise ValueError(f"Unknown replace_strategy: {self.replace_strategy}")

            # weights = self.confidence * (1.0 / (1 + (current_epoch - self.epoch)))
            # _, replace_indices = torch.topk(weights, batch_size, largest=False)

            self.memory[replace_indices] = features
            self.confidence[replace_indices] = confidence
            self.epoch[replace_indices] = current_epoch
    def sample_topk(self, k, current_epoch, cls_idx):
        """采样置信度和存储时间加权后的 k 个特征"""
        if self.ptr.item() == 0:
            return None
        # 确保 k 不超过当前记忆库中的特征数量
        k = min(k, self.ptr.item())
        # 确保 confidence 是张量并且是浮点类型
        confidence = self.confidence[: len(self.memory)].to(torch.float32)
        # 修改权重计算公式，避免时间差过小
        time_diff = current_epoch - self.epoch[: len(self.memory)]
        time_diff = torch.clamp(time_diff, min=1)  # 时间差至少为1，避免分母过小
        epsilon = 1e-10
        weights = confidence * (1.0 / (time_diff + epsilon))
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            info_if_main(
                f"[WARNING] Weights contain NaN or Inf in class {cls_idx}, Returning empty tensor."
            )
            return None
        if (weights < 0).any():
            info_if_main(
                f"[WARNING] Weights contain negative values in class {cls_idx}."
            )
            return None
        weights_sum = weights.sum()
        if weights_sum <= 0:
            info_if_main(f"[WARNING] Sum of weights is zero in class {cls_idx}.")
            return None
        # 归一化权重
        weights = weights / weights_sum
        # 尝试进行加权采样
        try:
            indices = torch.multinomial(
                weights, min(k, len(self.memory)), replacement=False
            )
        except RuntimeError as e:
            info_if_main(
                f"[ERROR] Multinomial sampling failed: {e}. Returning empty tensor."
            )
            return None
        return self.memory[indices]

    def get_center(self, device=None):
        """根据置信度加权计算特征中心"""
        if self.ptr.item() == 0:
            return None

        # 根据置信度加权平均
        weights = self.confidence[: len(self.memory)]
        weights = weights / weights.sum()  # 归一化
        center = torch.sum(
            self.memory[: len(self.memory)] * weights.unsqueeze(1),
            dim=0,
            keepdim=True,
        )
        if device is not None:
            return center.detach().to(device)

        return center.detach().to(self.memory.device)

    def __len__(self):
        return self.memory.size(0)


class MultiModalContrastLoss(nn.Module):
    def __init__(
        self,
        num_classes=9,
        vessel_idx=0,
        feat_dim=64,
        temp=0.5,
        num_queries=128,
        num_negatives=128,
        memobank_size=10000,
        high_conf_thresh=0.8,
        warmup_epochs=5,
        replace_strategy = 'cats'
    ):
        """
        Args:
            num_classes: 总类别数（含血管类）
            vessel_idx: 血管类索引
            feat_dim: 特征维度
            temp: 温度系数
            num_queries: 每类anchor数量
            num_negatives: 每个anchor的负样本数
            memobank_size: 记忆库容量
            high_conf_thresh: 高置信度阈值
        """
        super().__init__()
        self.num_classes = num_classes
        self.vessel_idx = vessel_idx
        self.feat_dim = feat_dim
        self.temp = temp
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.high_conf_thresh = high_conf_thresh
        self.warmup_epochs = warmup_epochs
        self.epoch_counter = 0
        # 初始化全局记忆库（每个类别独立）
        self.memobank = nn.ModuleList(
            [MemoryBank(memobank_size, feat_dim, replace_strategy=replace_strategy) for i in range(num_classes)]
        )

    def forward(self, features, preds, labels, vessel_mask, epoch):

        B, C, H, W, D = features.shape
        device = features.device

        # 这里 *不* detach()，保证 features 还能反向传播
        features = features.permute(0, 2, 3, 4, 1).reshape(-1, C)
        preds = preds.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        labels = labels.view(-1)
        vessel_mask = vessel_mask.view(-1)

        # 使用 softmax 计算概率分布
        prob = torch.softmax(preds, dim=1)
        epsilon = 1e-10
        prob = torch.clamp(prob, min=epsilon, max=1.0)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # 添加小常数
        # 处理熵为零或负数的情况
        entropy = torch.clamp(entropy, min=epsilon)  # 确保熵非负
        confidence = torch.exp(-entropy)
        # 裁剪置信度范围
        confidence = torch.clamp(confidence, min=1e-5, max=10)

        # 更新记忆库
        self.update_memobank(features, preds, labels, vessel_mask, epoch, confidence)

        if epoch < self.warmup_epochs:
            # info_if_main("[WARNING]  only warmup")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算对比损失
        loss = 0.0
        valid_classes = []
        for cls_idx in range(1, self.num_classes):
            cls_mask = labels == cls_idx
            if cls_mask.sum() == 0:
                continue

            indices = torch.nonzero(cls_mask, as_tuple=True)[0]
            cls_confidence = torch.index_select(confidence, dim=0, index=indices)

            # 动态调整采样数量
            actual_num_queries = min(len(indices), self.num_queries)
            # 按置信度选择 Top-K
            _, topk_idx = torch.topk(cls_confidence, actual_num_queries)
            sampled_indices = torch.index_select(indices, dim=0, index=topk_idx)

            # 这里不需要meta信息 转换为普通 Tensor  否则当样本不足时 会调用collate_fn 导致报错
            if isinstance(sampled_indices, MetaTensor):
                sampled_indices = sampled_indices.as_tensor()

            # 不足时重复采样
            if actual_num_queries < self.num_queries:
                repeat_times = (self.num_queries // actual_num_queries) + 1

                sampled_indices = sampled_indices.repeat(repeat_times)[
                    : self.num_queries
                ]

            anchors = torch.index_select(features, dim=0, index=sampled_indices)

            pos_center = self.memobank[cls_idx].get_center(device=device)

            neg_pool = self.build_negative_pool(cls_idx, device, epoch=epoch)

            loss += self.compute_pairwise_loss(anchors, pos_center, neg_pool)

            valid_classes.append(cls_idx)

        return (
            loss / len(valid_classes)
            if len(valid_classes) > 0
            else torch.tensor(0.0, device=device, requires_grad=True)
        )

    def update_memobank(self, features, preds, labels, vessel_mask, epoch, confidence):
        """更新记忆库，将每个批次中置信度最高的前5%的特征存入相应类别的存储空间"""
        # 确保 features 和 preds 是 MetaTensor 并转换为普通 Tensor
        # features = features.as_tensor().detach()  # 转换为普通 Tensor 并确保不带梯度
        # preds = preds.as_tensor().detach()  # 转换为普通 Tensor 并确保不带梯度
        features = features.as_tensor()  # 转换为普通 Tensor 并确保不带梯度
        preds = preds.as_tensor()  # 转换为普通 Tensor 并确保不带梯度
        labels = labels.as_tensor().long()  # 确保 labels 是 long 类型
        vessel_mask = vessel_mask.as_tensor().bool()  # 确保 vessel_mask 是布尔类型
        device = features.device
        for cls_idx in range(self.num_classes):
            if cls_idx == self.vessel_idx:
                cls_mask = vessel_mask
            else:
                cls_mask = labels == cls_idx

            # 选择当前类别的样本
            indices = torch.nonzero(cls_mask, as_tuple=True)[0].to(dtype=torch.long)
            if indices.numel() == 0:
                # info_if_main(f"No features for class {cls_idx} in this batch")
                continue

            # 选择置信度最高的前5%，至少保留一个样本
            num_samples_to_keep = max(
                int(indices.numel() * 0.05), 1
            )  # 至少保留一个样本
            selected_confidence = torch.index_select(confidence, dim=0, index=indices)
            sorted_indices = torch.argsort(
                -selected_confidence
            ).as_tensor()  # 按置信度降序排序
            top_indices = sorted_indices[:num_samples_to_keep]

            # 转换为普通 Tensor 进行索引操作
            selected_features = features[
                indices[top_indices]
            ].clone()  # 使用 clone() 确保梯度不回传
            selected_confidence = torch.index_select(
                selected_confidence, dim=0, index=top_indices
            ).clone()
            # 存入记忆库
            try:
                self.memobank[cls_idx].enqueue(
                    selected_features.to(device), selected_confidence.to(device), epoch
                )
            except Exception as e:
                # info_if_main(f"Error enqueuing features for class {cls_idx}: {e}")
                continue

    def build_negative_pool(self, current_cls, device, epoch):
        """构建负样本池：其他类中心 + 血管中心 + 采样"""
        # 从其他类别中采样置信度最高的特征（每个类别采样 16 个）
        neg_pool = []
        for cls_idx in range(self.num_classes):
            if cls_idx != current_cls:
                center = self.memobank[cls_idx].get_center()
                if center is None or torch.isnan(center).any():
                    continue
                # 确保采样数量不超过可用特征数量
                samples = self.memobank[cls_idx].sample_topk(15, epoch, cls_idx)
                if samples is None or torch.isnan(samples).any():
                    continue
                neg_pool.append(center.to(device))
                neg_pool.append(samples.to(device))
        times = 0
        while len(neg_pool) < (self.num_classes - 1) and times < 100:
            times += 1
            supp_cls_idx = torch.randint(0, self.num_classes, (1,)).item()
            if supp_cls_idx != current_cls:
                center = self.memobank[supp_cls_idx].get_center()
                supp_samples = self.memobank[supp_cls_idx].sample_topk(
                    15, epoch, cls_idx
                )
                if supp_samples is not None and not torch.isnan(center).any():
                    neg_pool.append(center.to(device))
                    neg_pool.append(supp_samples.to(device))
        if len(neg_pool) < (self.num_classes - 1):
            raise ValueError(
                f"Negative pool is not enough, only {len(neg_pool)} classes."
            )
        neg_pool = torch.cat(neg_pool, dim=0)
        neg_pool = F.normalize(neg_pool + 1e-6, dim=1)
        return neg_pool

    def compute_pairwise_loss(self, anchors, pos_center, neg_pool):

        # if torch.isnan(anchors).any():
        #     info_if_main("[ERROR] anchors contain NaN!")
        # if torch.isnan(pos_center).any():
        #     info_if_main("[ERROR] pos_center contain NaN!")
        # if torch.isnan(neg_pool).any():
        #     info_if_main("[ERROR] neg_pool contain NaN!")
        # if (
        #     torch.isnan(anchors).any()
        #     or torch.isnan(pos_center).any()
        #     or torch.isnan(neg_pool).any()
        # ):
        #     return torch.tensor(0.0, device=anchors.device, requires_grad=True)

        """对比损失计算核心"""
        anchors = F.normalize(anchors, dim=1, eps=1e-6)  # 使用 F.normalize 的 eps 参数
        pos_center = F.normalize(pos_center, dim=1, eps=1e-6)
        neg_pool = F.normalize(neg_pool, dim=1, eps=1e-6)

        # info_if_main(f"anchors: {anchors.device}, pos_center: {pos_center.device}, neg_pool: {neg_pool.device}")

        sim_pos = torch.sum(anchors * pos_center, dim=1)
        sim_neg = torch.mm(anchors, neg_pool.T)

        # 数值限制
        sim_pos = torch.clamp(sim_pos, min=-1.0, max=1.0)
        sim_neg = torch.clamp(sim_neg, min=-1.0, max=1.0)

        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temp
        labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)

        return F.cross_entropy(logits, labels)

    def save(self, path):
        """
        保存 memobank 到指定路径。
        Args:
            path (str): 保存文件的路径。
        """
        # 保存 memobank
        state = {
            "memobank": [bank.memory for bank in self.memobank],  # 每个类别的记忆库
            "memobank_confidence": [
                bank.confidence for bank in self.memobank
            ],  # 每个类别的置信度
        }
        # 使用高精度保存
        torch.save(state, path, _use_new_zipfile_serialization=True)
        print(f"Saved memobank to {path}")

    def load(self, path, device=None):
        """
        从指定路径加载 memobank。
        Args:
            path (str): 加载文件的路径。
            device (torch.device): 加载数据的目标设备（如 "cuda" 或 "cpu"）。
        """
        if device is None:
            device = next(self.memobank[0].parameters()).device  # 默认使用当前设备

        # 加载数据
        state = torch.load(path, map_location=device)

        # 恢复 memobank
        for i, bank in enumerate(self.memobank):
            bank.memory.copy_(state["memobank"][i])
            bank.confidence.copy_(state["memobank_confidence"][i])

        print(f"Loaded memobank from {path}")
