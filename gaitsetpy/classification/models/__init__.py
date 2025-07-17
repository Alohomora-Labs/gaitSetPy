from .random_forest import RandomForestModel
from .mlp import MLPModel
from .lstm import LSTMModel
from .bilstm import BiLSTMModel
from .gnn import GNNModel
from .cnn import CNNModel

def get_classification_model(name: str, **kwargs):
    """
    Factory function to get a classification model by name.

    Args:
        name (str): Name of the model. One of: 'random_forest', 'mlp', 'lstm', 'bilstm', 'gnn', 'cnn'.
        **kwargs: Model-specific parameters.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If the model name is not recognized.

    Example:
        model = get_classification_model('cnn', input_channels=20, num_classes=4)
    """
    name = name.lower()
    if name == 'random_forest':
        return RandomForestModel(**kwargs)
    elif name == 'mlp':
        return MLPModel(**kwargs)
    elif name == 'lstm':
        return LSTMModel(**kwargs)
    elif name == 'bilstm':
        return BiLSTMModel(**kwargs)
    elif name == 'gnn':
        return GNNModel(**kwargs)
    elif name == 'cnn':
        return CNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}. Supported: 'random_forest', 'mlp', 'lstm', 'bilstm', 'gnn', 'cnn'.")
