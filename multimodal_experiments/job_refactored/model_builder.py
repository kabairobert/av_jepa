from multimodal_experiments.job_refactored.architectures import DualPairModel, DiagonalPredictor, MLPPredictor
from multimodal_experiments.initial_trials.ssl_disentangling import FlowModel, build_flow_layers

def build_model_and_predictors(cfg, device):
    stage_count = cfg.model.get('stage_count', 6)
    num_dims = cfg.model.get('num_dims', 2)
    hidden_units = cfg.model.get('hidden_units', 128)
    
    model_a = FlowModel(build_flow_layers(stage_count=stage_count, num_dims=num_dims, hidden_units=hidden_units)).to(device)
    model_b = FlowModel(build_flow_layers(stage_count=stage_count, num_dims=num_dims, hidden_units=hidden_units)).to(device)
    dual_model = DualPairModel(model_a, model_b).to(device)
    
    predictor_type = cfg.model.get('predictor_type', 'none')
    if predictor_type == 'diagonal':
        predictor_a2b = DiagonalPredictor(num_dims).to(device)
        predictor_b2a = DiagonalPredictor(num_dims).to(device)
    elif predictor_type == 'mlp':
        hidden_dim = cfg.model.get('predictor_hidden_dim', 64)
        predictor_a2b = MLPPredictor(num_dims, hidden_dim).to(device)
        predictor_b2a = MLPPredictor(num_dims, hidden_dim).to(device)
    else:
        predictor_a2b = None
        predictor_b2a = None
        
    return {
        "dual_model": dual_model,
        "model_a": model_a,
        "model_b": model_b,
        "predictor_a2b": predictor_a2b,
        "predictor_b2a": predictor_b2a
    }
