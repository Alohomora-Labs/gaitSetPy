from .random_forest import RandomForestModel
from .mlp import MLPModel

# Optional PyTorch-dependent imports
try:
    from .lstm import LSTMModel
    from .bilstm import BiLSTMModel
    from .gnn import GNNModel
    from .cnn import CNNModel
    PYTORCH_AVAILABLE = True
except ImportError:
    LSTMModel = None
    BiLSTMModel = None
    GNNModel = None
    CNNModel = None
    PYTORCH_AVAILABLE = False

def get_classification_model(name: str, **kwargs):
    """
    Factory function to get a classification model by name.

    Args:
        name (str): Name of the model. One of: 'random_forest', 'mlp', 'lstm', 'bilstm', 'gnn', 'cnn'.
        **kwargs: Model-specific parameters.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If the model name is not recognized or PyTorch is not available for PyTorch models.

    Example:
        model = get_classification_model('cnn', input_channels=20, num_classes=4)
    """
    name = name.lower()
    if name == 'random_forest':
        return RandomForestModel(**kwargs)
    elif name == 'mlp':
        return MLPModel(**kwargs)
    elif name == 'lstm':
        if not PYTORCH_AVAILABLE or LSTMModel is None:
            raise ImportError("LSTM model requires PyTorch. Please install PyTorch to use this model.")
        return LSTMModel(**kwargs)
    elif name == 'bilstm':
        if not PYTORCH_AVAILABLE or BiLSTMModel is None:
            raise ImportError("BiLSTM model requires PyTorch. Please install PyTorch to use this model.")
        return BiLSTMModel(**kwargs)
    elif name == 'gnn':
        if not PYTORCH_AVAILABLE or GNNModel is None:
            raise ImportError("GNN model requires PyTorch. Please install PyTorch to use this model.")
        return GNNModel(**kwargs)
    elif name == 'cnn':
        if not PYTORCH_AVAILABLE or CNNModel is None:
            raise ImportError("CNN model requires PyTorch. Please install PyTorch to use this model.")
        return CNNModel(**kwargs)
    else:
        available_models = ['random_forest', 'mlp']
        if PYTORCH_AVAILABLE:
            available_models.extend(['lstm', 'bilstm', 'gnn', 'cnn'])
        raise ValueError(f"Unknown model name: {name}. Supported: {available_models}.")
