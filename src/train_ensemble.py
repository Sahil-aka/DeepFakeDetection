# src/train_ensemble.py
"""
Ensemble evaluation script.
Loads all three trained models (CNN, ResNeXt, LSTM) and evaluates them individually
and as an ensemble on the test set.
"""

import tensorflow as tf
from pathlib import Path
import sys

# ---------- GPU safety / memory settings ----------
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
except Exception as e:
    print("GPU configuration:", e)

# ---------- Config ----------
BASE_DIR = "../data/Dataset"
IMG_SIZE = (160, 160)
BATCH = 8

# ---------- Import ensemble model ----------
from model_ensemble import EnsembleModel

# ---------- Data pipeline ----------
def make_test_dataset(base_dir=BASE_DIR, img_size=IMG_SIZE, batch=BATCH):
    """Load and prepare test dataset."""
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base.resolve()}")
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base / "Test",
        image_size=img_size,
        batch_size=batch,
        shuffle=False
    )
    
    # Normalization
    def _norm(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    test_ds = test_ds.map(_norm, num_parallel_calls=4).prefetch(1)
    
    return test_ds

# ---------- Main evaluation ----------
def main():
    print("="*60)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*60)
    
    # Check if all models exist
    model_paths = {
        'cnn': 'models/basic_cnn_best.h5',
        'resnext': 'models/resnext_best.h5',
        'lstm': 'models/lstm_best.h5'
    }
    
    missing_models = []
    for name, path in model_paths.items():
        if not Path(path).exists():
            missing_models.append(f"{name}: {path}")
    
    if missing_models:
        print("\n❌ ERROR: The following model files are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease train all models first:")
        print("  python src/train_cnn.py")
        print("  python src/train_resnext.py")
        print("  python src/train_lstm.py")
        sys.exit(1)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_ds = make_test_dataset()
    print("Test dataset loaded!")
    
    # Create ensemble model
    print("\n" + "="*60)
    ensemble = EnsembleModel(model_paths=model_paths)
    
    # Display ensemble weights
    print("\nEnsemble weights:")
    for name, weight in ensemble.get_model_weights().items():
        print(f"  {name}: {weight:.3f}")
    
    # Evaluate ensemble
    print("\n" + "="*60)
    results = ensemble.evaluate(test_ds, use_weights=True)
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nIndividual Models:")
    print("-" * 40)
    for name in ['cnn', 'resnext', 'lstm']:
        if name in results:
            acc = results[name]['accuracy']
            loss = results[name]['loss']
            print(f"{name.upper():12s} - Accuracy: {acc:.4f} ({acc*100:.2f}%), Loss: {loss:.4f}")
    
    print("\nEnsemble Model:")
    print("-" * 40)
    if 'ensemble' in results:
        acc = results['ensemble']['accuracy']
        loss = results['ensemble']['loss']
        print(f"{'ENSEMBLE':12s} - Accuracy: {acc:.4f} ({acc*100:.2f}%), Loss: {loss:.4f}")
    
    # Find best individual model
    best_individual = max(['cnn', 'resnext', 'lstm'], 
                         key=lambda x: results[x]['accuracy'])
    best_acc = results[best_individual]['accuracy']
    ensemble_acc = results['ensemble']['accuracy']
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best individual model: {best_individual.upper()} ({best_acc*100:.2f}%)")
    print(f"Ensemble accuracy: {ensemble_acc*100:.2f}%")
    
    improvement = ensemble_acc - best_acc
    if improvement > 0:
        print(f"✅ Ensemble improves by: +{improvement*100:.2f}%")
    elif improvement < 0:
        print(f"⚠️  Ensemble is worse by: {improvement*100:.2f}%")
    else:
        print("➡️  Ensemble matches best individual model")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
