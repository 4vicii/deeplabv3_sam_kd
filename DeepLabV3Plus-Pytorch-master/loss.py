import torch
import torch.nn as nn
import torch.nn.functional as F


class PKDLoss(nn.Module):
    def __init__(self, student_feature_dim, teacher_feature_dim):
        super(PKDLoss, self).__init__()
        # 匹配通道大小
        self.adapter = nn.Conv2d(student_feature_dim, teacher_feature_dim, kernel_size=1, stride=1, padding=0)

    def norm(self, features: torch.Tensor) -> torch.Tensor:
        assert len(features.shape) == 4
        N, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        features = (features - mean) / (std + 1e-6)
        return features.reshape(N, H, W, C).permute(0, 3, 1, 2)

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        student_features = self.adapter(student_features)
        student_features = F.interpolate(student_features, size=teacher_features.shape[2:], mode='bilinear',
                                         align_corners=False)
        assert student_features.shape == teacher_features.shape
        norm_student = self.norm(student_features)
        norm_teacher = self.norm(teacher_features)

        loss = F.mse_loss(norm_student, norm_teacher)
        return loss


class LdisLoss(nn.Module):
    def __init__(self, student_feature_dim, teacher_feature_dim):
        super(LdisLoss, self).__init__()
        # 使用1x1卷积层来匹配通道数
        self.adapter = nn.Conv2d(student_feature_dim, teacher_feature_dim, kernel_size=1, stride=1, padding=0)

    def compute_relation_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the relation matrix for given features.
        """
        norm = F.normalize(features, p=2, dim=1)  # normalize features along the channel dimension
        relation_matrix = torch.matmul(norm, norm.transpose(1, 2))
        return relation_matrix

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        # 适配学生特征的通道数
        student_features = self.adapter(student_features)
        # 可以考虑插值来匹配空间维度，如果需要的话
        student_features = F.interpolate(student_features, size=teacher_features.shape[2:], mode='bilinear',
                                         align_corners=False)

        assert student_features.shape == teacher_features.shape

        R = self.compute_relation_matrix(teacher_features)
        R_prime = self.compute_relation_matrix(student_features)

        H, W = student_features.shape[2], student_features.shape[3]
        N = (H // 8) * (W // 8)
        loss = torch.sum(torch.abs(R - R_prime)) / (N * N)

        return loss


