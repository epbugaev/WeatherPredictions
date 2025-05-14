import torch
import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import random
import math
import traceback


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Mixture of Experts Implementation from EWMoE
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts."""

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert."""
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates."""
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s."""
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts."""

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts - use dropout instead of drop
        self.experts = nn.ModuleList([FeedForward(self.input_size, self.hidden_size, dropout=0.) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample."""
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates."""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating."""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        from torch.distributions.normal import Normal
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating."""
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """MoE forward pass with load balancing loss"""
        original_shape = x.shape
        
        # Reshape for MoE if needed
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])
            
        gates, load = self.noisy_top_k_gating(x, self.training)
        
        # # Calculate importance loss
        # importance = gates.sum(0)
        # loss = self.cv_squared(importance) + self.cv_squared(load)
        # loss *= loss_coef

        # # Print MoE routing statistics 
        # if not hasattr(self, 'print_counter'):
        #     self.print_counter = 0
        
        # Only print every 10 times to avoid log flood
        # if self.print_counter % 10 == 0:
        #     with torch.no_grad():
        #         expert_usage = load.detach().cpu().numpy()
        #         total_tokens = x.shape[0]
        #         routing_percent = 100 * expert_usage / total_tokens
        #         # # print(f"\nMoE Routing Statistics (tokens={total_tokens}):")
        #         # for i, percent in enumerate(routing_percent):
        #         #     print(f"Expert {i}: {percent:.2f}% tokens ({int(expert_usage[i])} tokens)")
        #         imbalance = self.cv_squared(importance).item()
        #         # print(f"Load imbalance: {imbalance:.4f}")
        
        # self.print_counter += 1

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        
        # Reshape back to original if needed
        if len(original_shape) > 2:
            y = y.reshape(*original_shape)
            
        return y
     
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1, use_moe=True, num_experts=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.use_moe = use_moe
        
        for _ in range(depth):
            # Choose between MoE or SwiGLU based on use_moe flag
            if use_moe:
                ff_module = MoE(input_size=dim, output_size=dim, num_experts=num_experts, hidden_size=mlp_dim, noisy_gating=True, k=2)
            else:
                ff_module = SwiGLU(dim, mlp_dim, drop=dropout)
                
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, ff_module),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)
    
class WeightedTokenMixing(nn.Module):
    """Weighted token mixing for improved long-range forecasting.
    This module allows the model to dynamically weight different temporal positions
    when making predictions, which is particularly useful for weather forecasting.
    """
    def __init__(self, dim, seq_len):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        
        # Weight matrix for token mixing (learnable)
        self.temporal_weights = nn.Parameter(torch.ones(seq_len, seq_len))
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        # Initialize with temporal decay pattern (more weight to recent tokens)
        with torch.no_grad():
            # Create decay pattern (newer tokens have more influence)
            decay_pattern = torch.exp(torch.linspace(0, -2, seq_len)).unsqueeze(0)
            decay_pattern = decay_pattern.repeat(seq_len, 1)
            
            # Make it causal for forecasting (can't see future tokens)
            mask = torch.tril(torch.ones(seq_len, seq_len))
            self.temporal_weights.data = decay_pattern * mask
        
    def forward(self, x):
        # x shape could be [batch, seq_len, dim] or [seq_len, dim]
        shape = x.shape
        
        # Handle both 2D and 3D inputs
        if len(shape) == 2:
            seq_len, dim = shape
            x = x.unsqueeze(0)  # Add batch dimension
            batch = 1
        else:
            batch, seq_len, dim = shape
            
        assert dim == self.dim, f"Expected dimension {self.dim}, got {dim}"
        
        # Project values
        values = self.value_proj(x)  # [batch, seq_len, dim]
        
        # Apply softmax to weights for each position
        mixing_weights = F.softmax(self.temporal_weights[:seq_len, :seq_len], dim=-1)  # [seq_len, seq_len]
        
        # Apply weighted mixing
        mixed = torch.matmul(mixing_weights, values)  # [batch, seq_len, dim]
        
        # Project back to original dimension
        output = self.output_proj(mixed)  # [batch, seq_len, dim]
        
        # Return in the same shape as input
        if len(shape) == 2:
            return output.squeeze(0)
        return output

class PredFormerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1, use_moe=True, num_experts=4):
        super(PredFormerLayer, self).__init__()

        self.temporal_transformer_first = GatedTransformer(dim, depth, heads, dim_head, 
                                                      mlp_dim, dropout, attn_dropout, drop_path, use_moe, num_experts)
        self.space_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                             mlp_dim, dropout, attn_dropout, drop_path, use_moe, num_experts)
        self.temporal_transformer_second = GatedTransformer(dim, depth, heads, dim_head, 
                                                       mlp_dim, dropout, attn_dropout, drop_path, use_moe, num_experts)
        
        # Add weighted token mixing for better temporal modeling
        self.temporal_mixer = WeightedTokenMixing(dim, 12)  # 12 for sequence length

    def forward(self, x):
        b, t, n, _ = x.shape        
        x_t, x_ori = x, x 
        
        # t branch (first temporal)
        x_t = rearrange(x_t, 'b t n d -> b n t d')
        x_t = rearrange(x_t, 'b n t d -> (b n) t d')
        x_t = self.temporal_transformer_first(x_t)
        
        # Safely apply weighted token mixing
        try:
            # Reshape for mixing
            bn, t, d = x_t.shape
            x_t_mixed = x_t.clone()  # Create a copy to avoid in-place modification issues
            
            # Process each sequence separately
            for i in range(bn):
                x_t_mixed[i] = self.temporal_mixer(x_t[i])
                
            x_t = x_t_mixed
        except Exception as e:
            # If there's an error, just skip the mixing
            print(f"Skipping temporal mixing due to: {e}")
        
        # s branch (space)
        x_ts = rearrange(x_t, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        x_ts = self.space_transformer(x_ts)
        
        # t branch (second temporal)
        x_tst = rearrange(x_ts, '(b t) n d -> b t n d', b=b)
        x_tst = rearrange(x_tst, 'b t n d -> b n t d')
        x_tst = rearrange(x_tst, 'b n t d -> (b n) t d')
        x_tst = self.temporal_transformer_second(x_tst)

        # ts output branch     
        x_tst = rearrange(x_tst, '(b n) t d -> b n t d', b=b)
        x_tst = rearrange(x_tst, 'b n t d -> b t n d', b=b) 
        
        return x_tst

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')
      
class PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.num_classes = self.num_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        self.use_moe = model_config.get('use_moe', True)  # Default to True if not specified
        self.num_experts = model_config.get('num_experts', 4)  # Default to 4 if not specified

        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )
        
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                               requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.blocks = nn.ModuleList([
            PredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, 
                             self.dim * self.scale_dim, self.dropout, self.attn_dropout, 
                             self.drop_path, self.use_moe, self.num_experts)
            for i in range(self.Ndepth)
        ])
        
        # Add an adaptive layer norm for better normalization across channels
        self.adaptive_ln = nn.LayerNorm(self.dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
            ) 
                
    def forward(self, x):
        B, T, C, H, W = x.shape
        assert C == self.num_channels
        # Patch Embedding для входа x
        x_embed = self.to_patch_embedding(x) # [B, T, num_patches, dim]
        
        x_combined = x_embed       

        # Position Embedding
        x_combined += self.pos_embedding.to(x.device)
        
        # PredFormer Encoder
        for idx, blk in enumerate(self.blocks):
            x_combined = blk(x_combined)
            
        # Apply adaptive layer norm
        x_combined = self.adaptive_ln(x_combined)
            
        # MLP head
        x = self.mlp_head(x_combined.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x


# Model configuration with MoE settings
model_config = {
    # image h w c
    'height': 128,
    'width': 256,
    'num_channels': 69,
    # video length in and out
    'pre_seq': 12,
    'after_seq': 12,
    # patch size
    'patch_size': 8,
    'dim': 128,  # Reduced from 256
    'heads': 4,  # Reduced from 8
    'dim_head': 32,
    # dropout
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'scale_dim': 2,  # Reduced from 4
    # depth
    'depth': 1,
    'Ndepth': 4,  # Reduced from 8
    # MoE settings
    'use_moe': True,
    'num_experts': 4,  # Using more experts with ContentAwareMoE
    'use_improved_model': True,  # Flag to use the improved model
    'path_to_constants': '/home/user/mamba_x_predformer/PredFormer/constants_1.40625deg.nc',
}

# Smaller config for CPU fallback
model_config_small = {
    # image h w c
    'height': 64,
    'width': 128,
    'num_channels': 69,
    # video length in and out
    'pre_seq': 12,
    'after_seq': 12,
    # patch size
    'patch_size': 8,
    'dim': 64, 
    'heads': 2,
    'dim_head': 32,
    # dropout
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'scale_dim': 2,
    # depth
    'depth': 1,
    'Ndepth': 2,
    # MoE settings
    'use_moe': True,
    'num_experts': 20,
    'use_improved_model': True,  # Flag to use the improved model
    'path_to_constants': '/home/user/mamba_x_predformer/PredFormer/constants_1.40625deg.nc',
}

if __name__ == "__main__":
    print("=== PredFormerMoE Implementation ===")
    
    # Force CPU with small config
    device = torch.device("cpu")
    print(f"Using device: {device}")
    config_to_use = model_config_small
    
    # Small dimensions for CPU
    batch_size = 1
    img_height = config_to_use['height']
    img_width = config_to_use['width']
    num_channels = config_to_use['num_channels']
    seq_len = config_to_use['pre_seq']
    
    print(f"Using configuration for {device}:")
    print(f"- Image dimensions: {img_height}x{img_width}")
    print(f"- Model dimension: {config_to_use['dim']}")
    print(f"- Number of blocks: {config_to_use['Ndepth']}")
    
    print("Initializing model...")
    model = PredFormer_Model(config_to_use)
    print("Model initialized successfully")
    
    # Count and print the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    try:
        print(f"Moving model to {device}...")
        model = model.to(device)
        print("Model moved to device successfully")
        
        # Create input with appropriate dimensions
        print("Creating test input tensor...")
        print(f"Input shape: [{batch_size}, {seq_len}, {num_channels}, {img_height}, {img_width}]")
        x = torch.rand(batch_size, seq_len, num_channels, img_height, img_width).to(device)
        
        # Forward pass
        print("Starting forward pass...")
        with torch.no_grad():
            output = model(x)
        print("Forward pass completed successfully")
        print(f"Output tensor shape: {output.shape}")
        
    except Exception as e:
        print(f"Error during model execution: {e}")
        traceback.print_exc()
        print("Model execution failed.")
