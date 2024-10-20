import torch
from torch import nn
import math


class ExponentialActivation(nn.Module):
    def __init__(self):
        super(ExponentialActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)
    
class NormalizationActivation(nn.Module):
    def __init__(self):
        super(NormalizationActivation, self).__init__()

    def forward(self, x):
        return nn.functional.normalize(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, positional_encoding_dim):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding_dim=positional_encoding_dim
        
        self.register_buffer('freqs', math.pi * (2.0 ** torch.arange(self.positional_encoding_dim, dtype=torch.float32).unsqueeze(0).unsqueeze(0)))
        
    def forward(self, xyz):
        xyz=xyz.unsqueeze(-1)*self.freqs
        
        xyz_sin = torch.sin(xyz).reshape(xyz.shape[0], 3*self.positional_encoding_dim)
        xyz_cos = torch.cos(xyz).reshape(xyz.shape[0], 3*self.positional_encoding_dim)
        
        return torch.cat([xyz_sin, xyz_cos], dim=-1).to(xyz_sin.device)

class GaussPredictorMLPSmall(nn.Module):
    def __init__(self, input_image_features_dim, hidden_dim, positional_encoding_dim, latent_codes, static_color):
        super(GaussPredictorMLPSmall, self).__init__()
        
        # Gaussian properties dimensions
        self.xyz_dim_in = 3*positional_encoding_dim*2
        self.xyz_dim_out = 3
        self.scaling_dim = 3
        self.rotation_dim = 4
        self.opacity_dim = 1
        self.color_dim_in = 3*positional_encoding_dim*2
        self.color_dim_out = 3
        self.static_color = static_color
        
        # Latent Codes
        self.latent_codes_enabled = latent_codes['enabled']
        self.latent_code_size = latent_codes['latent_code_size'] if self.latent_codes_enabled else 0
        
        # Input dim
        self.input_dim = self.xyz_dim_in + self.scaling_dim + self.rotation_dim + self.opacity_dim + self.color_dim_in + input_image_features_dim + self.latent_code_size
        
        # Hidden dim
        self.hidden_dim = hidden_dim
        
        # Model
        self.positional_encoding = PositionalEncoding(positional_encoding_dim)
        
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
        )
        
        # Heads for the different gaussian properties
        self.xyz_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.xyz_dim_out)
        )
        self.scaling_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.scaling_dim),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.rotation_dim),
            NormalizationActivation()
        )
        self.opacity_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.opacity_dim),
            nn.Sigmoid()
        )
        
        if not self.static_color:
            self.color_head = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=self.color_dim_out)
            )
        
        self.mlp1.apply(GaussPredictorMLPSmall.he_weight_init)
        #self.scaling_head.apply(GaussPredictorMLP.xavier_weight_init)
        #self.rotation_head.apply(GaussPredictorMLP.xavier_weight_init)
        #self.opacity_head.apply(GaussPredictorMLP.xavier_weight_init)
        
    def he_weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def xavier_weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, input):    
        mlp_input = [
            self.positional_encoding(input['xyz']),
            input['scaling'],
            input['rotation'],
            input['opacity'],
            self.positional_encoding(input['color']),
            input['features'],
        ]
        
        if self.latent_codes_enabled:
            mlp_input.append(input['latent_codes'])
        
        mlp_input = torch.cat(mlp_input, dim=-1).to(input['xyz'].device)
        mlp_output = self.mlp1(mlp_input)
        xyz_delta = (self.xyz_head(mlp_output) * 0.01)
        return {
            'xyz':  input['xyz'] + xyz_delta,
            'xyz_delta': xyz_delta,
            'scaling': self.scaling_head(mlp_output) * 0.01,
            'rotation': self.rotation_head(mlp_output) * 0.01,
            'opacity': self.opacity_head(mlp_output),
            'color': input['color'] if self.static_color else self.color_head(mlp_output)
        }