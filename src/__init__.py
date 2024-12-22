from .fashion_mnist import CNN, train, test, classes
from .model_variants import BaseCNN, DeepCNN, experiment
from .visualize import plot_training_process, plot_confusion_matrix, plot_sample_predictions

__all__ = [
    'CNN', 'train', 'test', 'classes',
    'BaseCNN', 'DeepCNN', 'experiment',
    'plot_training_process', 'plot_confusion_matrix', 'plot_sample_predictions'
]

__version__ = '0.1.0' 