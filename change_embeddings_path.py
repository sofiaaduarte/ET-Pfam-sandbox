"""
This script is designed to change the embeddings path in the config file for
each model in the models directory.

Parameters:
    -m, --models_path: Path to models directory containing subdirectories for each model
    -e, --embeddings_path: Path to new embeddings directory
    -f, --filter: Filter for models to change. Model names must contain this string

Usage example:
    python3 change_embeddings_path.py -m models/mini/ -e data/embeddings/esm2/
    -f esm2
"""

import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models_path", type=str, required=True, 
    help="Path to models directory containing subdirectories for each model")
parser.add_argument("-e", "--embeddings_path", type=str, required=True,
    help="Path to new embeddings directory")
parser.add_argument("-p", "--plm", type=str, required=False,
    help="Pre-trained language model (e.g. 'esm2', 'ptt5'). Default is 'esm2'",
    default="esm2")
parser.add_argument("-f", "--filter", type=str, required=False,
    help="Filter for models to change. Model names must contain this string"
                    " (e.g. 'esm')", default="")
args = parser.parse_args()

for model in os.listdir(args.models_path):
    model_path = os.path.join(args.models_path, model)
    # Only process directories
    if not os.path.isdir(model_path):
        continue
    if args.filter != "" and args.filter not in model:
        continue
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)

    # Change embeddings path
    print(f"Changing embeddings path for {model} from {args.embeddings_path} "
          f"to {args.embeddings_path}")
    config["emb_folder"] = args.embeddings_path

    # Set emb_dim based on the specified PLM
    if args.plm == "esm2":
        config["emb_dim"] = 1280
    elif args.plm == "ptt5":
        config["emb_dim"] = 1024
    else:
        raise ValueError(f"Unknown PLM: {args.plm}")
    print(f"Setting emb_dim to {config['emb_dim']}")

    # Save updated config
    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

print("Done")
