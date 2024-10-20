from model.networks.gauss_predictor_mlp import GaussPredictorMLP
from model.networks.gauss_predictor_mlp_small import GaussPredictorMLPSmall
from model.networks.gauss_predictor_mlp_large import GaussPredictorMLPLarge
from model.networks.gauss_predictor_mlp_improved import GaussPredictorMLPImproved

class ModelCollection():
    @classmethod
    def get_model(cls, config):
        
        model_name = config['model']['name']
        
        if model_name == 'gauss_predictor_mlp':
            return GaussPredictorMLP(**config['model']['model_parameters'])
        elif model_name == 'gauss_predictor_mlp_improved':
            return GaussPredictorMLPImproved(**config['model']['model_parameters'])
        elif model_name == 'gauss_predictor_mlp_small':
            return GaussPredictorMLPSmall(**config['model']['model_parameters'])
        elif model_name == 'gauss_predictor_mlp_large':
            return GaussPredictorMLPLarge(**config['model']['model_parameters'] )