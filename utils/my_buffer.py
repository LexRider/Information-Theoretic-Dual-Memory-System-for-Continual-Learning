import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from buffer import Buffer
import random

CLASSES_NUM = 10

class MNISTFeatureExtractor(nn.Module):
    def __init__(self, input_size=(28, 28), input_channels=1):
        super(MNISTFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels

    def forward(self, x, block_size=10):
        # 展平图像
        n = x.size(0)
        x_flattened = x.view(n, -1)

        return x_flattened

# class MNISTFeatureExtractor(nn.Module):
#     def __init__(self, model, input_size=(224, 224), input_channels=1):
#         super(MNISTFeatureExtractor, self).__init__()
#         self.input_size = input_size
#         self.input_channels = input_channels
#
#         # 修改模型的第一个卷积层以适应不同的输入通道数
#         layers = list(model.children())
#         first_conv_layer = layers[0]
#         self.first_conv_layer = nn.Conv2d(self.input_channels, first_conv_layer.out_channels,
#                                           kernel_size=first_conv_layer.kernel_size,
#                                           stride=first_conv_layer.stride,
#                                           padding=first_conv_layer.padding,
#                                           bias=first_conv_layer.bias)
#
#         # 构建新的特征提取器序列
#         self.features = nn.Sequential(
#             nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False),  # 动态调整输入大小
#             self.first_conv_layer,
#             *layers[1:-1]  # 去掉最后的全连接层
#         )
#
#     def forward(self, x, block_size=10):
#         n = x.size(0)
#         feature_list = []
#
#         # 分块处理
#         for i in range(0, n, block_size):
#             end = min(i + block_size, n)
#             x_block = x[i:end]
#             features_block = self.features(x_block)
#             features_block = features_block.view(features_block.size(0), -1)
#             feature_list.append(features_block)
#
#         # 将所有块的特征拼接起来
#         features = torch.cat(feature_list, dim=0)
#         return features

class CIFAR10FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(CIFAR10FeatureExtractor, self).__init__()
        # 加载预训练的 ResNet-18 模型
        self.model = models.resnet18(pretrained=pretrained)
        # 修改第一个卷积层，以适应 CIFAR-10 数据集 (3 通道, 32x32 图像)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 修改最大池化层，去掉它，因为 CIFAR-10 图像比较小
        self.model.maxpool = nn.Identity()
        # 去掉最后的全连接层
        self.model.fc = nn.Identity()

    def forward(self, x, block_size=10):
        n = x.size(0)
        feature_list = []

        # 分块处理
        for i in range(0, n, block_size):
            end = min(i + block_size, n)
            x_block = x[i:end]
            features_block = self.model(x_block)
            features_block = features_block.view(features_block.size(0), -1)
            feature_list.append(features_block)

        # 将所有块的特征拼接起来
        features = torch.cat(feature_list, dim=0)
        return features


class MyBuffer(Buffer):
    def __init__(self, buffer_size, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(buffer_size, device)
        self.device = device

        # 加载预训练的 ResNet-18 模型
        pretrained_model = models.resnet18(pretrained=True)
        MNISTfeature_extractor = MNISTFeatureExtractor(pretrained_model, input_channels=1)  # 确保输入通道为1
        CIFAR10feature_extractor = CIFAR10FeatureExtractor(pretrained=True)
        self.MNISTfeature_extractor = MNISTfeature_extractor.to(device)
        self.MNISTfeature_extractor.eval()
        self.CIFAR10feature_extractor = CIFAR10feature_extractor.to(device)
        self.CIFAR10feature_extractor.eval()

    def get_buffer_content(self):
        if self.labels is None:
            print("Buffer is empty.")
            return

        unique_labels, label_counts = self.labels.unique(return_counts=True)
        buffer_content = {label.item(): count.item() for label, count in zip(unique_labels, label_counts)}

        print("Buffer Content:")
        for label, count in buffer_content.items():
            print(f"Label {label}: {count} samples")
        print(f"data length: {self.examples.shape} | logits length: {self.logits.shape}")


    def get_replay_batch_from_buffer(self, sample_num):
        if len(self.examples) <= sample_num:
            return self.examples, self.labels, self.logits, self.task_labels, self.attention_maps

        unique_labels, label_counts = self.labels.unique(return_counts=True)
        label_count = unique_labels.size(0)
        samples_per_label = sample_num // label_count

        label_to_indices = {label.item(): (self.labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}

        selected_data = []
        selected_labels = []
        selected_logits = []
        selected_task_labels = []
        selected_attention_maps = []

        for label in unique_labels:
            indices = label_to_indices[label.item()]
            if len(indices) >= samples_per_label:
                selected_indices = indices[torch.randperm(len(indices))[:samples_per_label]]
            else:
                selected_indices = indices  # 样本不足时全选

            selected_data.append(self.examples[selected_indices])
            selected_labels.append(self.labels[selected_indices])
            if hasattr(self, 'logits'):
                selected_logits.append(self.logits[selected_indices])
            if hasattr(self, 'task_labels'):
                selected_task_labels.append(self.task_labels[selected_indices])
            # if hasattr(self, 'attention_maps') and self.attention_maps is not None:
            #     selected_attention_maps.extend([self.attention_maps[i] for i in selected_indices])

        X = torch.cat(selected_data)
        y = torch.cat(selected_labels)
        logits = torch.cat(selected_logits) if selected_logits else None
        task_labels = torch.cat(selected_task_labels) if selected_task_labels else None
        attention_maps = selected_attention_maps if selected_attention_maps else None

        return X, y, logits, task_labels, attention_maps

    def add_data(self, examples, labels=None, logits=None, task_labels=None, attention_maps=None, update_strategy=1,distance_method='cosine'):

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        if self.examples is None:
            if len(examples) > self.buffer_size:
                self.examples = examples[:self.buffer_size].to(self.device)
                if labels is not None:
                    self.labels = labels[:self.buffer_size].to(self.device)
                if logits is not None:
                    self.logits = logits[:self.buffer_size].to(self.device)
                if task_labels is not None:
                    self.task_labels = task_labels[:self.buffer_size].to(self.device)
            else:
                self.examples = examples.to(self.device)
                if labels is not None:
                    self.labels = labels.to(self.device)
                if logits is not None:
                    self.logits = logits.to(self.device)
                if task_labels is not None:
                    self.task_labels = task_labels.to(self.device)
        elif len(self.examples) + len(examples) <= self.buffer_size:
            self.examples = torch.cat((self.examples, examples.to(self.device)), dim=0)
            if labels is not None:
                self.labels = torch.cat((self.labels, labels.to(self.device)), dim=0)
            if logits is not None:
                self.logits = torch.cat((self.logits, logits.to(self.device)), dim=0)
            if task_labels is not None:
                self.task_labels = torch.cat((self.task_labels, task_labels.to(self.device)), dim=0)
        else:
            self.examples, self.labels, self.logits = self.buffer_update_fn(self.examples, self.labels, self.logits, examples, labels, logits, distance_method)
        self.num_seen_examples += len(examples)


    def buffer_update_fn(self, data_buffer, label_buffer, logits_buffer, data, labels, logits, distance_method):
        # 合并当前buffer和新batch的数据
        combined_data = torch.cat((data_buffer, data), dim=0)
        combined_labels = torch.cat((label_buffer, labels), dim=0)
        combined_logits = torch.cat((logits_buffer, logits), dim=0)

        # 确定新类别标签和已存在的标签
        unique_labels = combined_labels.unique()
        existing_labels = label_buffer.unique()
        new_labels = [label for label in unique_labels if label not in existing_labels]

        selected_indices = []

        # 对于每个已存在的标签，只更新这些标签的数据
        for label in existing_labels:
            if label in new_labels:
                continue
            indices = (combined_labels == label).nonzero(as_tuple=True)[0]
            features = self.get_features(combined_data[indices])
            if distance_method == 'cosine':
                distances = self.compute_average_cosine_distances(features, self.device)
            elif distance_method == 'js':
                distances = self.compute_average_js_divergences(features, self.device)
            else:
                raise ValueError("Invalid distance_method. Choose either 'cosine' or 'js'.")

            # 计算目标数量
            target_label_count = (self.buffer_size + len(unique_labels) - 1) // len(unique_labels)
            if len(indices) > target_label_count:
                probabilities = 1 - (distances / distances.sum())
                probabilities /= probabilities.sum()  # 确保概率总和为1
                indices_to_remove = torch.multinomial(probabilities, len(indices) - target_label_count,replacement=False)
                indices_to_keep = torch.tensor([i for i in range(len(indices)) if i not in indices_to_remove],device=self.device)

                # indices_to_remove = torch.topk(distances, len(indices) - target_label_count, largest=False).indices
                # indices_to_keep = torch.tensor([i for i in range(len(indices)) if i not in indices_to_remove],device=self.device)
            else:
                indices_to_keep = torch.arange(len(indices), device=self.device)

            selected_indices.extend(indices[indices_to_keep].tolist())

        # 对于新类别标签，应用原始逻辑
        for label in new_labels:
            indices = (combined_labels == label).nonzero(as_tuple=True)[0]
            features = self.get_features(combined_data[indices])
            if distance_method == 'cosine':
                distances = self.compute_average_cosine_distances(features, self.device)
            elif distance_method == 'js':
                distances = self.compute_average_js_divergences(features, self.device)
            else:
                raise ValueError("Invalid distance_method. Choose either 'cosine' or 'js'.")

            target_label_count = (self.buffer_size + len(unique_labels) - 1) // len(unique_labels)
            if len(indices) > target_label_count:
                probabilities = 1 - (distances / distances.sum())
                probabilities /= probabilities.sum()  # 确保概率总和为1
                indices_to_remove = torch.multinomial(probabilities, len(indices) - target_label_count,replacement=False)
                indices_to_keep = torch.tensor([i for i in range(len(indices)) if i not in indices_to_remove],device=self.device)

                # indices_to_remove = torch.topk(distances, len(indices) - target_label_count, largest=False).indices
                # indices_to_keep = torch.tensor([i for i in range(len(indices)) if i not in indices_to_remove],device=self.device)
            else:
                indices_to_keep = torch.arange(len(indices), device=self.device)

            selected_indices.extend(indices[indices_to_keep].tolist())

        # 更新缓存
        selected_indices = torch.tensor(selected_indices, device=self.device)
        updated_data_buffer = combined_data[selected_indices]
        updated_label_buffer = combined_labels[selected_indices]
        updated_logits_buffer = combined_logits[selected_indices]

        return updated_data_buffer, updated_label_buffer, updated_logits_buffer

    def get_features(self, samples):
        # 使用预训练模型提取样本特征
        with torch.no_grad():
            samples = samples.to(self.device)
            # print("getting features")
            features = self.MNISTfeature_extractor(samples)
        return features

    def compute_average_cosine_distances(self, features, device, batch_size=100):
      features = features.to(device)
      n, m = features.shape
      cosine_distance_list = []

      # 将特征按批次进行处理
      for start in range(0, n, batch_size):
          end = min(start + batch_size, n)
          features_batch = features[start:end]

          batch_distances = []
          for other_start in range(0, n, batch_size):
              other_end = min(other_start + batch_size, n)
              features_other_batch = features[other_start:other_end]

              # 计算当前批次的余弦相似度
              cosine_similarity_matrix = F.cosine_similarity(features_batch.unsqueeze(1), features_other_batch.unsqueeze(0), dim=2)
              cosine_distance_matrix = 1 - cosine_similarity_matrix

              if start == other_start:
                  # 遮掩对角线
                  mask = torch.eye(end - start, other_end - other_start, device=device).bool()
                  cosine_distance_matrix.masked_fill_(mask, 0)

              batch_distances.append(cosine_distance_matrix)

          # 将当前批次的所有距离矩阵拼接起来
          batch_avg_distance = torch.cat(batch_distances, dim=1).sum(dim=1) / (n - 1)
          cosine_distance_list.append(batch_avg_distance)

      # 将所有批次的平均距离拼接成最终结果
      avg_cosine_divergence = torch.cat(cosine_distance_list)
      return avg_cosine_divergence


    def compute_average_js_divergences(self, features, device):
        epsilon = 1e-10
        n, m = features.shape
        betas = torch.linspace(-6, 6, 100)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alphas = 1 - betas
        alphas_prod = torch.prod(alphas, dim=0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        diffused_features_means = features * alphas_bar_sqrt.to(device)
        diffused_features_covariances = (one_minus_alphas_bar_sqrt ** 2).to(device) * torch.eye(m, device=device)
        mid_means = (diffused_features_means.unsqueeze(1) + diffused_features_means.unsqueeze(0)) / 2
        mid_covariances = 0.5 * diffused_features_covariances
        cov_det = torch.linalg.det(diffused_features_covariances) + epsilon
        cov_inv = torch.linalg.inv(diffused_features_covariances)
        mid_cov_det = torch.linalg.det(mid_covariances) + epsilon
        mid_cov_inv = torch.linalg.inv(mid_covariances)

        part2 = torch.einsum('ij,ji->', mid_cov_inv, diffused_features_covariances)
        mu1_exp = diffused_features_means.unsqueeze(1)
        mu2_exp = diffused_features_means.unsqueeze(0)
        sigma2_inv_exp = mid_cov_inv.expand(n, n, m, m)
        sigma2_det_exp = mid_cov_det.expand(n, n)
        sigma1_det_exp = cov_det.expand(n, n)

        kl_p_m = self.kl_divergence(mu1_exp, mid_means, sigma2_inv_exp, sigma2_det_exp, sigma1_det_exp, m, part2,
                                    epsilon)
        kl_q_m = self.kl_divergence(mu2_exp, mid_means, sigma2_inv_exp, sigma2_det_exp, sigma1_det_exp, m, part2,
                                    epsilon)
        jsd = 0.5 * (kl_p_m + kl_q_m)
        mask = ~torch.eye(n, dtype=bool, device=device)
        jsd_valid = jsd[mask].view(n, n - 1)
        average_distances = jsd_valid.mean(dim=1)

        return average_distances

    def kl_divergence(self, mu1, mu2, sigma2_inv, sigma2_det, sigma1_det, m, precomputed_part2, epsilon):
        mu_diff = mu2 - mu1
        part1 = torch.log(sigma2_det / sigma1_det + epsilon)
        part3 = torch.einsum('...i,...ij,...j->...', mu_diff, sigma2_inv, mu_diff)
        return 0.5 * (part1 + precomputed_part2 + part3 - m)
