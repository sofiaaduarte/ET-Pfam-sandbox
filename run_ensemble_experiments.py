"""
This script runs multiple ensemble training and testing experiments by modifying
the ensemble configuration file and calling train_test_ensemble.py for each configuration.

Usage:
    python3 run_ensemble_experiments.py
"""
import json
import subprocess

def run_experiment(experiment_config, ensemble_config_path="config/ensemble.json", models_path="models/full/"):
    """
    Runs a single experiment by updating the ensemble config and calling train_test_ensemble.py
    
    Args:
        experiment_config (dict): Configuration for this experiment
        ensemble_config_path (str): Path to ensemble config file
    """
    # Update ensemble config file
    with open(ensemble_config_path, 'w') as f:
        json.dump(experiment_config, f, indent=4)
    
    # Build command - pass models_path and ensemble_config_path to train_test_ensemble.py
    cmd = [
        "python3",
        "train_test_ensemble.py",
        "-m", models_path,
        "-ec", ensemble_config_path,
    ]
    
    # Run the experiment
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"WARNING: Experiment failed with return code {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    
    # Define experiments to run
    experiments = [
        {
            "voting_strategy": "flatten_mlp",
            "use_bias": True,
            "hidden_size": 4096,
            "n_epochs": 2000,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4
        }, 
    ]
    
    print(f"\nTotal experiments to run: {len(experiments)}")
    
    # Run all experiments
    failed_experiments = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        return_code = run_experiment(exp_config)
        
        if return_code != 0:
            failed_experiments.append((i, exp_config['voting_strategy']))
    
    # Summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(experiments) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp_num, strategy in failed_experiments:
            print(f"  - Experiment {exp_num}: {strategy}")
    else:
        print("\nAll experiments completed successfully!")
    
    print("="*80)
