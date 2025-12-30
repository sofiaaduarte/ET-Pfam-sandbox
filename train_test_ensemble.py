"""
This script trains and tests ensemble models with configurable parameters.

The ensemble configuration file should contain:
    - voting_strategy: Type of ensemble ('weighted_model', 'weighted_families', 
                       'weighted_families_mlp', 'family_linear', 'family_mlp_linear',
                       'flatten_linear', 'flatten_mlp')
    - use_bias: Whether to use bias in the ensemble layer (for applicable strategies)
    - hidden_size: Size of hidden layer (for MLP-based strategies)
    - learning_rate: Learning rate for ensemble training
    - n_epochs: Number of training epochs

Note: exp_name is automatically generated from the parameters.

Parameters:
    -m, --models_path: Path to the directory containing the models to ensemble.
    -c, --config_path: Path to the base configuration file (.json).
    -ec, --ensemble_config: Path to the ensemble configuration file (.json).
    -p, --partition: Dataset partition to test on (default: 'test').
    -o, --output_path: Path to save the results (optional).
    
Usage example:
    python3 train_test_ensemble.py -m models/mini/ -ec config/ensemble.json
"""
import os
import argparse
import torch as tr
import json
import time
from pathlib import Path
from src.ensemble import EnsembleModel
from src.centered_window_test import centered_window_test
from src.sliding_window_test import sliding_window_test
from src.utils import load_config

tr.multiprocessing.set_sharing_strategy('file_system')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--models_path", type=str, required=False, 
                        help="Path to the models to ensemble",
                        default="models/mini/")
    parser.add_argument("-c","--config_path", type=str, required=False,
                        help="Path to the base config file (.json)",
                        default="config/base.json")
    parser.add_argument("-ec","--ensemble_config", type=str, required=False,
                        help="Path to the ensemble config file (.json)",
                        default="config/ensemble.json")
    parser.add_argument("-p", "--partition", type=str, required=False,
                        help="Dataset partition to test on (default: 'test')",
                        default='test')     
    parser.add_argument("-o", "--output_path", type=str, required=False,
                        help="Path to save the results",
                        default=None)
    
    args = parser.parse_args()
    return args

def run_ensemble_train_test(
        models_path: str,            # Path to ensemble model directories
        config: dict,                # Configuration dictionary
        voting_strategy: str,        # Selected voting strategy
        output_path: str,            # Directory to store results
        exp_name: str = None,        # Experiment name for saving ensemble weights
        hidden_size: int = 4,        # Hidden size for MLP-based strategies
        use_bias: bool = True,       # Whether to use bias in the ensemble layer
        learning_rate: float = 0.01, # Learning rate for ensemble training
        n_epochs: int = 500,         # Number of training epochs
        partition: str = 'test'      # Dataset partition to test on
        ):
    """Trains and tests an ensemble model"""
    width = os.get_terminal_size().columns

    print(f"\nTraining and testing ensemble with voting strategy: {voting_strategy}")

    # Initialize ensemble model
    ensemble = EnsembleModel(
        models_path, 
        config, 
        voting_strategy,
        exp_name=exp_name,
        hidden_size=hidden_size,
        use_bias=use_bias
    )
    
    # Train the ensemble
    print("\n" + "-" * width)
    print(f"Training ensemble (lr={learning_rate}, epochs={n_epochs})...")
    ensemble.fit(learning_rate=learning_rate, n_epochs=n_epochs)

    # Test the ensemble with centered window method
    print("\n" + "-" * width)
    print("Running centered window test...")
    CwS = centered_window_test(config, ensemble, output_path, is_ensemble=True,
                         voting_strategy=voting_strategy, partition=partition)

    # Test the ensemble with sliding window method
    print("\n" + "-" * width)
    print("Running sliding window test...")
    _, SwA, SwC = sliding_window_test(config, ensemble, output_path, is_ensemble=True,
                                       partition=partition)

    return CwS, SwA, SwC

if __name__ == "__main__":
    args = parser()
    width = os.get_terminal_size().columns
    
    # Load configurations
    config = load_config(args.config_path)
    ensemble_config_path = args.ensemble_config
    with open(ensemble_config_path, 'r') as f:
        ensemble_config = json.load(f)
    
    # Extract ensemble parameters
    voting_strategy = ensemble_config.get('voting_strategy', 'family_linear')
    use_bias = ensemble_config.get('use_bias', True)
    hidden_size = ensemble_config.get('hidden_size', 64)
    learning_rate = ensemble_config.get('learning_rate', 0.01)
    n_epochs = ensemble_config.get('n_epochs', 500)
    partition = args.partition
    
    # Generate exp_name based on parameters
    bias_str = "bias" if use_bias else "nobias"
    time_str = time.strftime("%d%m%Y-%H%M%S")
    exp_name = f"h{hidden_size}_lr{learning_rate}_ep{n_epochs}_{bias_str}_{partition}_{time_str}"
    exp_name_folder = f"{voting_strategy}_{exp_name}"

    # Validate voting strategy
    valid_strategies = [
        'weighted_model', 'weighted_families', 'weighted_families_mlp',
        'family_linear', 'family_mlp_linear', 'flatten_linear', 'flatten_mlp'
    ]
    
    if voting_strategy not in valid_strategies:
        raise ValueError(f"Invalid voting strategy: {voting_strategy}. " \
                         f"Valid strategies: {valid_strategies}")
    
    # Set output path
    if args.output_path is None:
        dataset = config.get('dataset', 'mini')
        args.output_path = f"results/{dataset}/{exp_name_folder}"
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    # Save ensemble config to results path
    ensemble_config_save_path = os.path.join(args.output_path, "ensemble_config.json")
    with open(ensemble_config_save_path, 'w') as f:
        json.dump(ensemble_config, f, indent=4)
    print(f"\nSaved ensemble configuration to: {ensemble_config_save_path}")
    
    print("\n" + "=" * width)
    print("ENSEMBLE CONFIGURATION")
    print("=" * width)
    print(f"Voting strategy: {voting_strategy}")
    print(f"Use bias: {use_bias}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training epochs: {n_epochs}")
    print(f"Experiment name: {exp_name}")
    print(f"Models path: {args.models_path}")
    print(f"Output path: {args.output_path}")
    print(f"Test partition: {args.partition}")
    
    # Train and test the ensemble
    print("\n" + "=" * width)
    print("TRAINING AND TESTING ENSEMBLE")
    print("=" * width)
    
    CwS, SwA, SwC = run_ensemble_train_test(
        args.models_path,
        config,
        voting_strategy,
        args.output_path,
        exp_name=exp_name,
        hidden_size=hidden_size,
        use_bias=use_bias,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        partition=args.partition
    )
    
    # Print final results
    print("\n" + "=" * width)
    print("FINAL RESULTS")
    print("=" * width)
    print(f"Centered Window Score: {CwS:.4f}")
    print(f"Sliding Window Accuracy: {SwA:.4f}")
    print(f"Sliding Window Coverage: {SwC:.4f}")
    print("=" * width)
