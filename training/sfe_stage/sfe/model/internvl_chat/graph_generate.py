import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
# from discription_generate import generate_description
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch.optim import Adam
import json
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from tqdm import tqdm

global_model = None
global_tokenizer = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def des_gene(image_path):
    res = generate_description(image_path)
    return res

def build_transform(input_size):
    """
    创建图像预处理 Transform
    :param input_size: 目标输入尺寸 (H, W)
    :return: 预处理 Transform
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # 确保是 RGB 格式
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),  # 调整尺寸
        T.ToTensor(),  # 转换为 Tensor
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # 归一化
    ])
    return transform

def dynamic_preprocess(image, image_size=448, max_num=12, split_factor=2):
    """
    先按照 `target_ratio` 进行一次切片，再对每个 Patch 进行 `split_factor × split_factor` 切割
    :param image: 输入 PIL 图像
    :param image_size: Patch 大小
    :param max_num: 最大 Patch 数量
    :param split_factor: 每个 Patch 进一步切割成 split_factor x split_factor 个子 Patch
    :return: 切分后的 patch 图像列表, (rows, cols) -> 最终 Patch 网格大小
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算最合适的切片数量
    target_ratios = [(i, j) for i in range(1, max_num + 1) for j in range(1, max_num + 1) if i * j <= max_num]
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - (r[0] / r[1])))

    # 计算目标尺寸
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]

    # 重新调整图像大小
    resized_img = image.resize((target_width, target_height))

    # **计算第一次切割的 `grid_size`**
    rows = target_ratio[1]
    cols = target_ratio[0]

    # **计算最终的 `grid_size`**
    final_rows = rows * split_factor
    final_cols = cols * split_factor
    grid_size = (final_rows, final_cols)

    # 第一次切割
    processed_images = []
    for i in range(target_ratio[0]):  # 横向
        for j in range(target_ratio[1]):  # 纵向
            box = (
                i * image_size, j * image_size,
                (i + 1) * image_size, (j + 1) * image_size
            )
            split_img = resized_img.crop(box)

            # **二次切割（每个 Patch 切成 split_factor × split_factor）**
            for si in range(split_factor):
                for sj in range(split_factor):
                    small_box = (
                        si * (image_size // split_factor), sj * (image_size // split_factor),
                        (si + 1) * (image_size // split_factor), (sj + 1) * (image_size // split_factor)
                    )
                    small_patch = split_img.crop(small_box)
                    processed_images.append(small_patch)

    print(f"Image {image.size} -> {resized_img.size} -> {len(processed_images)} patches")
    return processed_images, grid_size

def extract_id_image(file_path):
    id_image_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            id_image_list.append({"id": data["id"], "image": data["image"]})
    return id_image_list

def load_image(image_path, input_size=448, max_num=12):
    """
    读取并预处理图像，返回 image_tensor
    :param image_path: 图片路径
    :param input_size: Patch 大小
    :param max_num: 最大 Patch 数量
    :return: image_tensor, shape (B, 3, H, W)
    """
    image = Image.open(image_path).convert('RGB')  # 读取图像
    transform = build_transform(input_size=input_size)

    # 动态切片
    images, grid_size = dynamic_preprocess(image, image_size=input_size, max_num=max_num)

    # 转换为张量
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)  # (B, 3, H, W)

    return pixel_values, grid_size

def load_jsonl_data(jsonl_file, base_path):
    """
    读取 JSONL 文件，并返回 (image_path, description) 对列表。

    :param jsonl_file: JSONL 文件路径
    :param base_path: 根路径，用于拼接图片路径
    :return: [(image_path, description), ...] 形式的列表
    """
    image_desc_pairs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            image_path = os.path.join(base_path, data["image"])  # 拼接完整路径
            description = data["description"]
            image_desc_pairs.append((image_path, description))
    return image_desc_pairs

class VisionFeatureMLP(nn.Module):
    """
    用于将 ViT 提取的特征映射到 4096 维度
    """
    def __init__(self, input_dim=1024, output_dim=4096):
        super(VisionFeatureMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),  # 先映射到 2048
            nn.ReLU(),
            nn.Linear(2048, output_dim),  # 最终映射到 4096
            nn.ReLU()
        )

    def forward(self, vit_embeds):
        """
        :param vit_embeds: [B, N, 1024]
        :return: [B, 4096]
        """
        vit_embeds = vit_embeds[:, 1:, :]  # 移除 CLS Token (ViT 的第一个 token)
        vit_embeds = vit_embeds.mean(dim=1)  # 取所有 Patch 的均值 [B, 1024]
        vit_embeds = self.mlp(vit_embeds)  # [B, 4096]
        return vit_embeds


class GraphBuilder:
    def __init__(self, vision_model, language_model, tokenizer):
        """
        图构建类
        :param vision_model: InternVL 视觉模型
        :param language_model: InternVL 语言模型
        :param tokenizer_name: 语言模型的 tokenizer 名称
        """
        self.vision_model = vision_model   # InternVL 视觉编码器
        self.language_model = language_model  # InternVL 语言模型
        self.tokenizer = tokenizer  # 语言模型的 tokenizer
        self.device = vision_model.device  # 设备

    def build_tg(self, text):
        """
        构建文本图（TG）: 按 `\n` 分割文本，每个句子作为结点
        :param text: 输入文本
        :return: NetworkX 图对象
        """
        G = nx.Graph()
        
        # 1. 解析文本 & 按 `\n` 分割成单位
        text_units = text.strip().split("\n")  # 每行一个单位
        
        # 2. Tokenize 并获取嵌入
        sentence_embeddings = []
        for idx, sentence in enumerate(text_units):
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(self.device)  # 转移到 GPU
            
            with torch.no_grad():
                embeddings = self.language_model.get_input_embeddings()(input_ids)  # (Seq_len, hidden_dim)

            sentence_embedding = embeddings.mean(dim=1).squeeze(0)  # 取平均表示
            sentence_embeddings.append(sentence_embedding)

            # 添加结点
            G.add_node(idx, text=sentence, embedding=sentence_embedding)

        # 3. 计算句子间的相似度 & 添加边
        num_nodes = len(text_units)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = F.cosine_similarity(G.nodes[i]['embedding'], G.nodes[j]['embedding'], dim=0)
                G.add_edge(i, j, weight=sim.item())

        return G

    def build_vg(self, image_tensor, grid_size):
        """
        构建视觉图（VG）：Patch 作为结点，构造局部连接图
        :param image_tensor: 预处理后的输入图像张量 [B, 3, H, W]，每个 B 代表一个 Patch
        :param grid_size: Patch 网格大小 (rows, cols)
        :return: NetworkX 局部连接图
        """
        G = nx.Graph()
        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=image_tensor.to(self.device),
                output_hidden_states=True,
                return_dict=True
            )

        vit_embeds = vision_outputs.last_hidden_state
        vision_mlp = VisionFeatureMLP(input_dim=1024, output_dim=4096).to(dtype=torch.bfloat16, device=self.device)
        vit_embeds = vision_mlp(vit_embeds)  # [B, 4096]

        B = vit_embeds.shape[0]
        rows, cols = grid_size  # 获取行数和列数

        # 添加 Patch 结点
        for i in range(B):
            G.add_node(i, embedding=vit_embeds[i].detach().clone())

        # 只连接相邻的 Patch
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c  # 计算 Patch 索引
                if r > 0:
                    G.add_edge(idx, (r - 1) * cols + c)  # 连接上方
                if r < rows - 1:
                    G.add_edge(idx, (r + 1) * cols + c)  # 连接下方
                if c > 0:
                    G.add_edge(idx, r * cols + (c - 1))  # 连接左侧
                if c < cols - 1:
                    G.add_edge(idx, r * cols + (c + 1))  # 连接右侧

        return G


import torch
import torch.nn.functional as F

def compute_importance(vg_embeddings, tg_embeddings, vg_edge_index, top_k=5):
    """
    计算 VG 结点内部传播后的特征，再与 TG 结点计算相似度，返回 top-k 的 VG 结点及其对应的 top-k 的 TG 结点，拼接后作为 Soft Prompt。
    :param vg_embeddings: [num_vg_nodes, 4096] VG 结点特征
    :param tg_embeddings: [num_tg_nodes, 4096] TG 结点特征
    :param vg_edge_index: [2, num_edges] VG 的边索引
    :param top_k: 选择 top_k 个 VG 结点及其对应的 TG 结点
    :return: 拼接后的 Soft Prompt [top_k * 2, 4096]
    """
    num_vg_nodes = vg_embeddings.shape[0]
    num_tg_nodes = tg_embeddings.shape[0]
    
    # **第一步: 计算 VG 结点的邻居特征均值**
    edge_src, edge_dst = vg_edge_index  # 获取边的起点和终点索引
    
    # 计算相邻结点之间的相似度作为边权重
    edge_weights = F.cosine_similarity(vg_embeddings[edge_src], vg_embeddings[edge_dst], dim=-1)  # [num_edges]
    
    # 归一化边权重，使其在 0~1 之间（避免极端值）
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-6)
    
    # 计算邻居特征的加权平均
    neighbor_mean = torch.zeros_like(vg_embeddings)  # [num_vg_nodes, 4096]
    
    for i in range(num_vg_nodes):
        neighbors = (edge_src == i).nonzero().squeeze(1)
        if len(neighbors) > 0:
            # 计算邻居的加权均值
            weighted_sum = torch.sum(vg_embeddings[edge_dst[neighbors]] * edge_weights[neighbors].unsqueeze(-1), dim=0)
            neighbor_mean[i] = weighted_sum / edge_weights[neighbors].sum()

    # **第二步: 计算每个 VG 结点与 TG 结点的相似度**
    # 将 VG 结点与 TG 结点的相似度计算出来
    sim_matrix = F.cosine_similarity(vg_embeddings.unsqueeze(1), tg_embeddings.unsqueeze(0), dim=-1)  # [num_vg_nodes, num_tg_nodes]
    
    # **第三步: 结合邻居特征和 VG 结点特征**
    combined_embeddings = (vg_embeddings + neighbor_mean) / 2  # [num_vg_nodes, 4096]
    
    # **第四步: 计算最终重要性并选出 top_k 的 VG 结点**
    importance = torch.norm(combined_embeddings, dim=-1) * sim_matrix.max(dim=-1).values  # [num_vg_nodes]
    top_vg_indices = torch.topk(importance, top_k, largest=True).indices  # 选出最重要的 top_k VG 结点

    # **第五步: 选出与 top_k VG 结点最相似的 TG 结点**
    top_tg_indices = torch.topk(sim_matrix[top_vg_indices], 1, largest=True).indices  # 对应的 TG 结点
    
    # **第六步: 使用 torch.gather 获取 TG 结点特征**
    # 使用 top_tg_indices 获取 tg_embeddings 中的相应特征，并保持其维度为 [top_k, 4096]
    
    tg_selected = tg_embeddings[top_tg_indices.view(-1)]  # [top_k * top_k, 4096]
    tg_selected = tg_selected.view(1, 1, -1)  # [top_k, top_k, 4096]
    
    # **第七步: 拼接 VG 结点和 TG 结点的特征**
    # 取出 top_k 个 VG 结点的特征，并将它们与 TG 结点拼接

    soft_prompt = torch.cat([combined_embeddings[top_vg_indices], tg_selected.view(-1, 4096)], dim=0)  # [top_k + 3, 4096]
    
    return soft_prompt


class GraphContrastiveModel(nn.Module):
    """
    计算 TG (Text Graph) 和 VG (Vision Graph) 的嵌入，并进行图对比学习
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GraphContrastiveModel, self).__init__()
        
        # 文本图 (TG) 的 GCN
        self.gcn_tg1 = GCNConv(in_dim, hidden_dim)
        self.gcn_tg2 = GCNConv(hidden_dim, out_dim)

        # 视觉图 (VG) 的 GCN
        self.gcn_vg1 = GCNConv(in_dim, hidden_dim)
        self.gcn_vg2 = GCNConv(hidden_dim, out_dim)

        # MLP 进一步对齐 TG 和 VG 的特征
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )


    def forward(self, tg_x, tg_edge_index, vg_x, vg_edge_index, temperature=0.1):
        """
        计算节点级别的对比损失 (InfoNCE loss)
        :param tg_embeddings: 文本图节点嵌入 [num_nodes, dim]
        :param vg_embeddings: 视觉图节点嵌入 [num_nodes, dim]
        :param temperature: 温度参数 τ
        """
        tg_x = F.relu(self.gcn_tg1(tg_x, tg_edge_index))
        tg_embeddings = self.gcn_tg2(tg_x, tg_edge_index)
        vg_x = F.relu(self.gcn_tg1(vg_x, vg_edge_index))
        vg_embeddings = self.gcn_tg2(vg_x, vg_edge_index)
        # 归一化嵌入
        tg_embeddings = F.normalize(tg_embeddings, p=2, dim=1)
        vg_embeddings = F.normalize(vg_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵 
        # sim[i][j] = exp(θ(ui,vj)/τ)
        sim_matrix = torch.mm(tg_embeddings, vg_embeddings.t()) / temperature
        sim_matrix = torch.clamp(sim_matrix, min=-50, max=50) 
        
        # 创建正样本对的标签 (对角线位置)
        # 论文中提到 ui 和 vi 是代表相同对象的节点对
        labels = torch.arange(tg_embeddings.size(0)) % vg_embeddings.size(0)
        labels = labels.to(tg_embeddings.device)
        # 计算 InfoNCE loss
        # 分子: exp(θ(ui,vi)/τ)
        # 分母: exp(θ(ui,vi)/τ) + Σ exp(θ(ui,vk)/τ) where k≠i
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

class GraphContrastiveTrainer:
    def __init__(self, graph_builder, model, save_path, lr=1e-4):
        """
        :param graph_builder: 用于构造 TG 和 VG 的 GraphBuilder
        :param model: GraphContrastiveModel
        :param lr: 学习率
        """
        self.graph_builder = graph_builder
        self.model = model.to("cuda")
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.save_path = save_path


    def train_step(self, image_tensor, text):
        """
        单步训练
        """
        self.optimizer.zero_grad()

        TG = self.graph_builder.build_tg(text)
        VG = self.graph_builder.build_vg(image_tensor)

        # 获取 PyG 格式的输入
        
        data_tg = from_networkx(TG)
        data_vg = from_networkx(VG)
        tg_x, tg_edge_index = data_tg.embedding, data_tg.edge_index
        vg_x, vg_edge_index = data_vg.embedding, data_vg.edge_index

        tg_x = F.normalize(tg_x, p=2, dim=1)
        vg_x = F.normalize(vg_x, p=2, dim=1)
        tg_x = tg_x.to("cuda:0", dtype=torch.bfloat16)
        vg_x = vg_x.to("cuda:0", dtype=torch.bfloat16)
        tg_edge_index = tg_edge_index.to("cuda", dtype=torch.int64)
        vg_edge_index = vg_edge_index.to("cuda", dtype=torch.int64)
        print(tg_x)
        print(vg_x)

        # 计算 loss
        loss = self.model(tg_x, tg_edge_index, vg_x, vg_edge_index)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataset, epochs=10):
        """
        训练 GCL，并保存模型
        :param dataset: `[(image_tensor, text), (image_tensor, text), ...]`
        :param epochs: 训练轮次
        """
        best_loss = float("inf")  # 记录最小 loss
        for epoch in range(epochs):
            total_loss = 0

            # tqdm进度条，显示 loss
            progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
            for image_tensor, text in progress_bar:
                loss = self.train_step(image_tensor, text)
                progress_bar.set_postfix({"Batch Loss": f"{loss:.4f}"})
                total_loss += loss
                
                # 更新进度条，显示Batch Loss
                progress_bar.set_postfix({"Batch Loss": f"{loss:.4f}"})

            # 计算Avg Loss
            avg_loss = total_loss / len(dataset)
            print(f"\nEpoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

            # 如果 loss 下降，保存模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model()
                print(f"Model saved at {self.save_path} (Best Loss: {best_loss:.4f})")

        print("Training complete.")


    def save_model(self):
        """
        保存模型到 self.save_path
        """
        torch.save(self.model.state_dict(), self.save_path)


def generate_soft_prompt(img_path, dis, model, tokenizer):
    """ 生成 soft prompt，并避免 OOM """
    vision_model = model.vision_model
    language_model = model.language_model
    graph_builder = GraphBuilder(vision_model, language_model, tokenizer)

    image, grid_size = load_image(img_path)
    VG = graph_builder.build_vg(image, grid_size)
    TG = graph_builder.build_tg(dis)

    data_tg = from_networkx(TG)
    data_vg = from_networkx(VG)
    
    tg_x, tg_edge_index = data_tg.embedding, data_tg.edge_index
    vg_x, vg_edge_index = data_vg.embedding, data_vg.edge_index

    soft_prompt = compute_importance(vg_x, tg_x, vg_edge_index, top_k=5)

    return soft_prompt

if __name__ == '__main__':
    model_path = "/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/pretrained_models/InternVL2-8B"
    base_path = "/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2"
    data_path = "/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/dataset/internvl_format_latest/internvl2_train.jsonl"
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    vision_model = model.vision_model
    language_model = model.language_model

    graph_builder = GraphBuilder(vision_model, language_model, tokenizer)

    data = extract_id_image(data_path)
    image_p = os.path.join(base_path, data[0]["image"])
    dis = generate_description(image_p)
    image, grid_szie = load_image(image_p)
    VG = graph_builder.build_vg(image, grid_szie)
    TG = graph_builder.build_tg(dis)
    
    data_tg = from_networkx(TG)
    data_vg = from_networkx(VG)
    tg_x, tg_edge_index = data_tg.embedding, data_tg.edge_index
    vg_x, vg_edge_index = data_vg.embedding, data_vg.edge_index
    print(tg_x.shape)
    print(vg_x.shape)
    print(tg_edge_index.shape)
    print(vg_edge_index.shape)
    soft_prompt = compute_importance(vg_x, tg_x, vg_edge_index, top_k=3)
    print("Soft Prompt Shape:", soft_prompt.shape)  # 应该是 [top_k, 4096]