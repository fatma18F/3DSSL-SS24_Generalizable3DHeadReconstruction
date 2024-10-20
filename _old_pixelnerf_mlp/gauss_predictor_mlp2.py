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

class MultiResolutionHashEncoding(nn.Module):
    def __init__(self, L, F, log2_T):
        super(MultiResolutionHashEncoding, self).__init__()
        self.L = L  # Number of levels
        self.F = F  # Number of features per level
        self.log2_T = log2_T  # Log2 of the number of hash table entries per level

    def forward(self, x):
        # x is expected to be of shape [N, 3], where N is the number of points
        features = []
        for l in range(self.L):
            # Hash encoding logic
            hashed = (x * 2**l).int() % (2**self.log2_T)
            features.append(hashed.float())
        return torch.cat(features, dim=-1)
    
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

class GaussPredictorMLP(nn.Module):
    def __init__(self, input_image_features_dim, hidden_dim=512, positional_encoding_dim=20):
        super(GaussPredictorMLP, self).__init__()
        # Gaussian properties dimensions
        self.xyz_dim_in = 3*positional_encoding_dim*2
        self.xyz_dim_out = 3
        self.scaling_dim = 3
        self.rotation_dim = 4
        self.opacity_dim = 1
        self.color_dim = 3
        
        # Input dim
        self.input_dim = self.xyz_dim_in + self.scaling_dim + self.rotation_dim + self.opacity_dim + self.color_dim + input_image_features_dim
        
        # Hidden dim
        self.hidden_dim = hidden_dim
        
        # Model
        self.positional_encoding = PositionalEncoding(positional_encoding_dim)
        L = 4
        F=2
        self.encoder = MultiResolutionHashEncoding(L, F, log2_T=19)
        self.shit=L*hidden_dim - hidden_dim # 7680
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim+self.input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
        )
        
        # Heads for the different gaussian properties
        self.xyz_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim+self.shit+self.input_dim, out_features=hidden_dim), #
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.xyz_dim_out)
        )
        self.scaling_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim+self.shit+self.input_dim, out_features=hidden_dim),#
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.scaling_dim),
            #ExponentialActivation()
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim+self.shit+self.input_dim, out_features=hidden_dim), #
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.rotation_dim),
            NormalizationActivation()
        )
        self.opacity_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim+self.shit+self.input_dim, out_features=hidden_dim), #
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.opacity_dim),
            nn.Sigmoid()
        )
        self.color_head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim+self.input_dim+self.shit, out_features=hidden_dim),#
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.color_dim)
        )
        
        self.mlp1.apply(GaussPredictorMLP.he_weight_init)
        self.mlp2.apply(GaussPredictorMLP.he_weight_init)
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
        mlp_input = torch.cat([
            self.positional_encoding(input['xyz']),
            input['scaling'],
            input['rotation'],
            input['opacity'],
            input['color'].squeeze(1),
            input['features'],
        ], dim=-1).to('cuda')
        print('mlp_input shape:', mlp_input.shape)#[5143, 910] 
        mlp_output = self.mlp1(mlp_input)
        print('mlp_output shape:', mlp_output.shape)#[5143, 512] 
        encoded_x = self.encoder(mlp_output)
        print('encoded_x shape:', encoded_x.shape) #[5143, 8192]
        mlp_output=encoded_x

        #mlp_output = torch.cat([mlp_input, mlp_output], dim=-1).to('cuda')
        #mlp_output = self.mlp2(mlp_output)
       
        
        mlp_output = torch.cat([mlp_input, mlp_output], dim=-1).to('cuda')
        print('mlp_output shape:', mlp_output.shape)#5143, 9102]
        
        
        return {
            'xyz':  input['xyz'] + (self.xyz_head(mlp_output) * 0.01),
            'scaling': self.scaling_head(mlp_output)*0.01,
            'rotation': self.rotation_head(mlp_output) * 0.01,
            'opacity': self.opacity_head(mlp_output),
            'color': self.color_head(mlp_output).unsqueeze(1)
        }