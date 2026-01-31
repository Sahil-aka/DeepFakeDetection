# src/model_ensemble.py
import tensorflow as tf
import numpy as np
from pathlib import Path

class EnsembleModel:
    """
    Ensemble model that combines predictions from CNN, ResNeXt, and LSTM models.
    Supports simple averaging and weighted averaging strategies.
    """
    
    def __init__(self, model_paths=None, weights=None):
        """
        Initialize ensemble model.
        
        Args:
            model_paths: Dict with keys 'cnn', 'resnext', 'lstm' and values as paths to .h5 files
            weights: Optional dict with same keys, values are weights for weighted averaging
                    If None, uses simple averaging (equal weights)
        """
        if model_paths is None:
            # Default paths
            model_paths = {
                'cnn': 'models/basic_cnn_best.h5',
                'resnext': 'models/resnext_best.h5',
                'lstm': 'models/lstm_best.h5'
            }
        
        self.model_paths = model_paths
        self.models = {}
        self.weights = weights
        
        # Load models
        self._load_models()
        
        # Set default weights if not provided
        if self.weights is None:
            self.weights = {'cnn': 1/3, 'resnext': 1/3, 'lstm': 1/3}
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def _load_models(self):
        """Load all models from disk."""
        print("Loading models for ensemble...")
        for name, path in self.model_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            print(f"  Loading {name} from {path}")
            self.models[name] = tf.keras.models.load_model(path)
        print("All models loaded successfully!")
    
    def predict(self, x, use_weights=True):
        """
        Make ensemble predictions.
        
        Args:
            x: Input data (images)
            use_weights: If True, uses weighted averaging. If False, simple averaging.
        
        Returns:
            Ensemble predictions (averaged)
        """
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            predictions[name] = model.predict(x, verbose=0)
        
        # Combine predictions
        if use_weights:
            ensemble_pred = sum(predictions[name] * self.weights[name] 
                              for name in self.models.keys())
        else:
            ensemble_pred = sum(predictions.values()) / len(predictions)
        
        return ensemble_pred, predictions
    
    def evaluate(self, dataset, use_weights=True):
        """
        Evaluate ensemble on a dataset.
        
        Args:
            dataset: tf.data.Dataset
            use_weights: If True, uses weighted averaging
        
        Returns:
            Dictionary with metrics for each model and ensemble
        """
        all_labels = []
        all_preds = {name: [] for name in self.models.keys()}
        all_preds['ensemble'] = []
        
        print("Evaluating ensemble...")
        for x_batch, y_batch in dataset:
            # Get ensemble predictions
            ensemble_pred, individual_preds = self.predict(x_batch, use_weights=use_weights)
            
            # Store predictions
            all_labels.extend(y_batch.numpy())
            all_preds['ensemble'].extend(ensemble_pred.flatten())
            for name, pred in individual_preds.items():
                all_preds[name].extend(pred.flatten())
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        for name in all_preds.keys():
            all_preds[name] = np.array(all_preds[name])
        
        # Calculate metrics
        results = {}
        for name, preds in all_preds.items():
            # Binary predictions (threshold at 0.5)
            binary_preds = (preds > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(binary_preds == all_labels)
            
            # Calculate binary cross-entropy loss
            epsilon = 1e-7  # To avoid log(0)
            preds_clipped = np.clip(preds, epsilon, 1 - epsilon)
            loss = -np.mean(all_labels * np.log(preds_clipped) + 
                           (1 - all_labels) * np.log(1 - preds_clipped))
            
            results[name] = {
                'accuracy': accuracy,
                'loss': loss
            }
        
        return results
    
    def get_model_weights(self):
        """Return current ensemble weights."""
        return self.weights.copy()
    
    def set_model_weights(self, weights):
        """
        Set new weights for ensemble.
        
        Args:
            weights: Dict with keys 'cnn', 'resnext', 'lstm'
        """
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
