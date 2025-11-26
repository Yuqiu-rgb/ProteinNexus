# cga_utils.py
import torch
from kingdon import Algebra

# 1. 定义三维共形几何代数 (G_4,1)
# 符号差: 3个正 (e1, e2, e3), 1个负 (en), 1个正 (ep)
# 我们将使用标准的零基约定，其中 en 和 ep 构成 eo 和 einf。
# Kingdon 使用 p, q, r 符号差 (正, 负, 零)。G_4,1 的 p=4, q=1, r=0。
# 基向量将是 e1, e2, e3, e4 (平方为1), e5 (平方为-1)
alg = Algebra(4, 1, 0)
e1, e2, e3, e4, e5 = alg.blades['e1'], alg.blades['e2'], alg.blades['e3'], alg.blades['e4'], alg.blades['e5']

# 从 e4 (ep) 和 e5 (en) 定义零基
eo = 0.5 * (e5 - e4)
einf = e5 + e4

# 定义欧几里得子空间的伪标量
I_3 = e1 * e2 * e3


def up_project_points(points: torch.Tensor):
    """
    将一批3D点上投影到5D CGA零向量。
    Args:
        points: 形状为 [..., 3] 的张量，表示3D坐标。
    Returns:
        一个 kingdon MultiVector 对象，表示该批次的CGA点。
        其系数张量的形状将是 [..., 32]。
    """
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("输入点张量的形状必须是 [..., 3]")

    # 多重向量将具有与输入点相同的批处理形状
    batch_shape = points.shape[:-1]

    # 为每个分量创建多重向量
    x_mv = alg.vector(e1=points[..., 0], e2=points[..., 1], e3=points[..., 2])

    # 计算共形因子的平方范数
    sq_norm = torch.sum(points ** 2, dim=-1)

    # 构建完整的CGA点
    # X = x + 0.5 * |x|^2 * einf + eo
    # 我们需要将 eo 和 einf 广播到批处理形状
    eo_broadcast = alg.vector(eo.values.expand(*batch_shape, -1))
    einf_broadcast = alg.vector(einf.values.expand(*batch_shape, -1))

    # 标量 sq_norm 需要整形以与多重向量进行广播
    sq_norm_shaped = sq_norm.view(*batch_shape, 1)

    cga_points = x_mv + 0.5 * sq_norm_shaped * einf_broadcast + eo_broadcast
    return cga_points


def motor_from_params(bivector_coeffs: torch.Tensor, translation_vec: torch.Tensor):
    """
    从二重矢量（用于旋转）和矢量（用于平移）参数构建一个运动子。
    Args:
        bivector_coeffs: 形状为 [..., 3, C] 的张量，用于 e23, e31, e12 分量。
        translation_vec: 形状为 [..., 3, C] 的张量，用于 e1, e2, e3 分量。
    Returns:
        一个 kingdon MultiVector 对象，表示该批次的运动子。
    """
    # 确保通道在最后以便 kingdon 处理
    bivector_coeffs = bivector_coeffs.permute(*range(bivector_coeffs.ndim - 2), -1, -2)  # [..., C, 3]
    translation_vec = translation_vec.permute(*range(translation_vec.ndim - 2), -1, -2)  # [..., C, 3]

    # 1. 创建用于旋量部分的二重矢量
    # B = b_x e23 + b_y e31 + b_z e12
    B = alg.bivector(e23=bivector_coeffs[..., 0], e31=bivector_coeffs[..., 1], e12=bivector_coeffs[..., 2])

    # 2. 创建旋量 R = exp(-B/2)
    # 对于小角度，exp(A) 约等于 1 + A。我们可以直接学习系数。
    # 一个更稳定的方法是使用公式 R = cos(|B|/2) - sin(|B|/2) * B_normalized
    b_norm_sq = torch.sum(bivector_coeffs ** 2, dim=-1, keepdim=True)
    b_norm = torch.sqrt(b_norm_sq + 1e-8)
    b_normalized = B / b_norm

    cos_half_norm = torch.cos(b_norm / 2)
    sin_half_norm = torch.sin(b_norm / 2)

    Rotor = cos_half_norm + sin_half_norm * b_normalized

    # 3. 创建平移子 T = 1 - 0.5 * d * einf
    d = alg.vector(e1=translation_vec[..., 0], e2=translation_vec[..., 1], e3=translation_vec[..., 2])
    Translator = 1 - 0.5 * d * einf

    # 4. 运动子是几何积 M = T * R
    Motor = Translator * Rotor
    return Motor


def sandwich_product(versor, mv):
    """
    计算夹积 V * X * ~V。
    Args:
        versor: 算子（例如，运动子，旋量）。一个 kingdon MultiVector。
        mv: 要被变换的多重向量。一个 kingdon MultiVector。
    Returns:
        变换后的多重向量。
    """
    return versor * mv * ~versor


# cegan_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from cga_utils import alg, up_project_points, motor_from_params, sandwich_product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CEGALayer(nn.Module):
    def __init__(self, in_channels, out_channels, attention_heads=4, attention_temp=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_heads = attention_heads
        self.attention_temp = attention_temp

        # 用于头拆分的总通道数
        self.total_out_channels = out_channels * attention_heads

        # 用于创建消息的 MLP。输入是两个连接的多重向量。
        # 每个多重向量有32个分量。
        self.mlp_m = nn.Linear(in_channels * 2 * 32, self.total_out_channels * 32)

        # 用于预测运动子参数的 MLP。
        # 输入：来自两个多重向量的不变特征。
        # 我们将使用标量部分和向量/二重矢量部分的范数。
        # (s_i, s_j, |v_i|, |v_j|, |b_i|, |b_j|) -> 6 * in_channels
        self.mlp_motor = nn.Sequential(
            nn.Linear(in_channels * 6, 128),
            nn.ReLU(),
            # 输出：每个头3个二重矢量系数 + 3个平移矢量系数
            nn.Linear(128, self.attention_heads * 6)
        )

        # 用于值投影的线性层（注意力的一部分）
        self.lin_v = nn.Linear(in_channels * 32, self.total_out_channels * 32)

        # 用于最终更新的 MLP，包括残差连接
        self.mlp_update = nn.Linear(in_channels * 32, out_channels * 32)
        self.norm = nn.LayerNorm([32, out_channels])

    def forward(self, h, coords, adj_matrix):
        """
        CEGA 层的前向传播。
        Args:
            h: 输入的多重向量特征 (kingdon MultiVector)。
            coords: C-alpha 坐标，形状为。
            adj_matrix: 邻接矩阵，形状为。
        """
        B, N, C_in = h.shape, h.shape[1], self.in_channels

        # 为 MLP 扁平化多重向量： ->
        h_flat = h.values.permute(0, 1, 3, 2).reshape(B, N, C_in * 32)

        # 1. 预测运动子 (规范连接)
        # 提取用于运动子预测的不变特征
        s_i = h.grade(0).values.squeeze(-2)  #
        v_i_norm = torch.norm(h.grade(1).values[..., :3, :], dim=-2)  #
        b_i_norm = torch.norm(h.grade(2).values, dim=-2)  #

        s_j, v_j_norm, b_j_norm = s_i, v_i_norm, b_i_norm

        inv_i = torch.cat([s_i, v_i_norm, b_i_norm], dim=-1).unsqueeze(2).expand(-1, -1, N, -1)
        inv_j = torch.cat([s_j, v_j_norm, b_j_norm], dim=-1).unsqueeze(1).expand(-1, N, -1, -1)

        motor_params = self.mlp_motor(torch.cat([inv_i, inv_j], dim=-1))  #
        motor_params = motor_params.reshape(B, N, N, self.attention_heads, 6)

        bivector_coeffs = motor_params[..., :3].unsqueeze(-2).permute(0, 1, 2, 3, 5, 4)  #
        translation_vec = motor_params[..., 3:].unsqueeze(-2).permute(0, 1, 2, 3, 5, 4)  #

        motors = motor_from_params(bivector_coeffs, translation_vec)  # 多重向量

        # 2. 内在几何注意力
        cga_points = up_project_points(coords)  # 多重向量
        # 扩展以进行成对计算
        P_i = alg.multivector(values=cga_points.values.unsqueeze(2).expand(-1, -1, N, -1))
        P_j = alg.multivector(values=cga_points.values.unsqueeze(1).expand(-1, N, -1, -1))

        # 内积给出负的距离平方
        attention_logits = (P_i | P_j).grade(0).values.squeeze(-1) / self.attention_temp  #

        # 应用邻接矩阵掩码
        attention_logits.masked_fill_(adj_matrix.unsqueeze(1).expand(-1, N, -1) == 0, -1e9)
        attention_weights = F.softmax(attention_logits, dim=-1)  #

        # 3. 使用平行输运进行消息传递
        # 投影到每个头的值 (V)
        v_flat = self.lin_v(h_flat).reshape(B, N, self.attention_heads, self.out_channels, 32)
        v = alg.multivector(values=v_flat.permute(0, 1, 2, 4, 3))  #

        # 扩展以进行消息传递
        v_j = alg.multivector(values=v.values.unsqueeze(1).expand(-1, N, -1, -1, -1, -1))  #

        # 将消息从 j 平行输运到 i
        # motors:, v_j: -> 广播
        transported_messages = sandwich_product(motors, v_j)  #

        # 4. 聚合与更新
        # 使用注意力权重聚合消息
        # attn_weights:, 需要为头和通道扩展
        attn_expanded = attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  #

        aggregated_messages = torch.sum(transported_messages.values * attn_expanded, dim=2)  #

        # 连接头
        aggregated_flat = aggregated_messages.permute(0, 1, 3, 2, 4).reshape(B, N, 32, self.total_out_channels)

        # 投影回 out_channels 维度（或使用线性层）
        # 为简单起见，我们可以对头进行平均
        aggregated_mean = torch.mean(aggregated_flat, dim=-1)  #

        # 残差连接和最终更新
        h_res_flat = self.mlp_update(h_flat)
        h_res = alg.multivector(values=h_res_flat.reshape(B, N, self.out_channels, 32).permute(0, 1, 3, 2))

        h_out = h_res + alg.multivector(values=aggregated_mean.permute(0, 1, 2, 3).reshape(B, N, 32, self.out_channels))
        h_out.values = self.norm(h_out.values)

        return h_out


# cegan_model.py
import torch
import torch.nn as nn
from cga_utils import alg, up_project_points
from cegan_layer import CEGALayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CEGANet(nn.Module):
    def __init__(self, saprot_dim, channels, n_layers, n_neighbors=30, n_heads=4):
        super().__init__()
        self.saprot_dim = saprot_dim
        self.channels = channels
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors

        # 特征提升层
        self.lift_saprot = nn.Linear(saprot_dim, channels)

        # CEGA 层堆栈
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(CEGALayer(channels, channels, attention_heads=n_heads))

        # 读出层 (分类头)
        # 输入: 标量, |vec|, |bivec|, |trivec| 范数 -> C * 4
        self.readout = nn.Sequential(
            nn.Linear(channels * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, saprot_embed, coords):
        """
        整个 CEGANet 模型的前向传播。
        Args:
            saprot_embed: SaProt 嵌入，形状为。
            coords: C-alpha 坐标，形状为。
        """
        B, N, _ = saprot_embed.shape

        # 1. 图构建 (k-NN)
        dist_matrix = torch.cdist(coords, coords)
        _, nn_indices = torch.topk(dist_matrix, self.n_neighbors, dim=-1, largest=False)
        adj_matrix = torch.zeros(B, N, N, device=device)
        adj_matrix.scatter_(2, nn_indices, 1)

        # 2. 将特征提升到 CGA 多重向量
        # 提升标量特征
        s_part = self.lift_saprot(saprot_embed)

        # 将坐标上投影到 CGA 点
        cga_points = up_project_points(coords)

        # 初始化多重向量特征
        # h0 是一个 kingdon MultiVector，带有一批值
        h0_values = torch.zeros(B, N, 32, self.channels, device=device)

        # 填充标量部分 (0级, 索引0)
        h0_values[..., 0, :] = s_part

        # 从上投影的点填充向量部分
        # cga_points.values 是
        point_values = cga_points.values.unsqueeze(-1).expand(-1, -1, -1, self.channels)
        h0_values += point_values

        h = alg.multivector(values=h0_values)

        # 3. CEGA 层传播
        for layer in self.layers:
            h = layer(h, coords, adj_matrix)

        # 4. 读出与预测
        s_final = h.grade(0).values.squeeze(-2)
        v_final_norm = torch.norm(h.grade(1).values[..., :3, :], dim=-2)  # 欧几里得部分
        b_final_norm = torch.norm(h.grade(2).values, dim=-2)
        t_final_norm = torch.norm(h.grade(3).values, dim=-2)

        invariant_features = torch.cat([s_final, v_final_norm, b_final_norm, t_final_norm], dim=-1)

        logits = self.readout(invariant_features).squeeze(-1)

        return torch.sigmoid(logits)


# ==============================================================================
# 示例用法
# ==============================================================================

