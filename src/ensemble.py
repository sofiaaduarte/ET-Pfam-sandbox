import os
import torch as tr
import numpy as np
from torch import nn
from tqdm import tqdm
from src.basemodel import BaseModel
from src.dataset import PFamDataset
from src.utils import load_config, predict
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class WeightedModelLayer(nn.Module):
    """This is the original weighted model layer, where each model has a single weight."""
    def __init__(self, num_models):
        super(WeightedModelLayer, self).__init__()
        self.weights = nn.Parameter(tr.rand(num_models))

    def forward(self, x): # x: (num_models, batch, num_classes)
        return tr.sum(x * self.weights.view(-1, 1, 1), dim=0)

class WeightedFamiliesLayer(nn.Module):
    """This is the original weighted family layer, where each model has a weight per PF."""
    def __init__(self, num_models, num_classes):
        super(WeightedFamiliesLayer, self).__init__()
        self.weights = nn.Parameter(tr.rand(num_models, num_classes))

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Weights need to be broadcasted over the batch dimension
        return tr.sum(x * self.weights.view(self.weights.shape[0], 1, -1), dim=0)

class WeightedFamiliesLayerPytorch(nn.Module):
    """PyTorch implementation of WeightedFamiliesLayer using nn.Linear."""
    def __init__(self, num_models, num_classes):
        super(WeightedFamiliesLayerPytorch, self).__init__()
        self.linear = nn.Linear(num_models, num_classes, bias=False)

    def forward(self, x): # x: (num_models, batch, num_classes)
        z = x.permute(1, 2, 0)  # (batch, num_classes, num_models)
        z = self.linear(z)      # (batch, num_classes, num_classes)
        return z

class EnsembleModel(nn.Module):
    def __init__(self, models_path, config, voting_strategy, ensemble_weights_path=None, 
                 exp_name=None):
        super(EnsembleModel, self).__init__()

        # Load model paths
        model_dirs = [os.path.join(models_path, d) for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        
        self.emb_path = config['emb_path']
        self.data_path = config['data_path']
        self.voting_strategy = voting_strategy
        self.path = models_path
        self.weights_file = None

        self.available_weighted_strategies = {
            'weighted_model': WeightedModelLayer,
            'weighted_families': WeightedFamiliesLayer,
            'weighted_families_pytorch': WeightedFamiliesLayerPytorch
        }

        # Load categories
        cat_path = os.path.join(self.data_path, "categories.txt")
        with open(cat_path, 'r') as f:
            categories = [item.strip() for item in f]
        self.categories = categories

        # Initialize the ensemble of models
        self.models = nn.ModuleList()
        self.model_configs = []

        # Sort model directories to ensure consistent order
        model_dirs.sort()

        # Load each model's configuration and weights
        for model_dir in model_dirs:
            # Load the config.json to get the parameters
            config_path = os.path.join(model_dir, 'config.json')
            config = load_config(config_path)

            lr = config['lr']
            batch_size = config['batch_size']
            win_len = config['window_len']
            device = config['device']

            self.model_configs.append({
                'lr': lr,
                'batch_size': batch_size,
                'win_len': win_len
            })

            # Load the model weights
            weights_path = os.path.join(model_dir, 'weights.pk')
            print("loading weights from", model_dir)
            model = BaseModel(len(categories), lr=lr, device=device)
            model.load_state_dict(tr.load(weights_path))
            model.eval()
            self.models.append(model)
        
        # Initialize voting layer based on strategy
        self.voting_layer, self.weights_file = self._initialize_voting_layer(
            len(model_dirs), len(self.categories), ensemble_weights_path, exp_name
        )

    def fit(self):
        if self.voting_strategy in ['weighted_model', 'weighted_families', 'weighted_mlp']:
            # Collect predictions from each model
            all_preds = []
            for i, net in enumerate(self.models):
                config = self.model_configs[i]
                dev_data = PFamDataset(
                    f"{self.data_path}dev.csv",
                    self.emb_path,
                    self.categories,
                    win_len=config['win_len'],
                    is_training=False
                )
                dev_loader = tr.utils.data.DataLoader(dev_data, batch_size=config['batch_size'], num_workers=config.get("nworkers", 1))

                with tr.no_grad():
                    _, _, pred, ref, *_ = net.pred(dev_loader)
                    all_preds.append(pred)
            stacked_preds = tr.stack(all_preds)

            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam(self.voting_layer.parameters(), lr=0.01)

            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = self.voting_layer(stacked_preds)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            tr.save(self.voting_layer.state_dict(), self.weights_file)
            print(f"Saved voting layer weights to {self.weights_file}")

    def forward(self, batch):
        pred, _ = self.pred(batch)
        return pred

    def pred(self, partition='test'):
        # Predicts using the centered window method on the specified dataset.
        all_preds = []

        for i, net in enumerate(self.models):
            # Load the model's configuration and dataset
            config = self.model_configs[i]
            test_data = PFamDataset(
                f"{self.data_path}{partition}.csv",
                self.emb_path,
                self.categories,
                win_len=config['win_len'],
                is_training=False
            )
            test_loader = tr.utils.data.DataLoader(test_data, 
                                                   batch_size=config['batch_size'], 
                                                   num_workers=config.get("nworkers", 1))
            net_preds = []

            # Predict using the model
            with tr.no_grad():
                test_loss, test_errate, pred, *_ = net.pred(test_loader)
                net_preds.append(pred)
            print(f"win_len = {config['win_len']} - lr = {config['lr']} - test_loss {test_loss:.5f} - test_errate {test_errate:.5f}")
            net_preds = tr.cat(net_preds)
            all_preds.append(net_preds)

        stacked_preds = tr.stack(all_preds)
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)
        return preds, preds_bin

    def pred_sliding(self, emb, step=4, use_softmax=False):
        all_preds = []
        all_centers = []
        
        for i, net in enumerate(self.models):
            config = self.model_configs[i]
            net_preds = []
            centers, pred = predict(net, emb, config['win_len'], 
                                    use_softmax=use_softmax, step=step)
            net_preds.append(pred)
            all_preds.append(tr.cat(net_preds))
            all_centers.append(centers)

        for c in all_centers:
            if not np.allclose(c, all_centers[0]):
                raise ValueError("Model predictions have misaligned window centers.")

        stacked_preds = tr.stack(all_preds) 
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)
        return centers, preds.cpu().detach()

    def _initialize_voting_layer(self, num_models, num_classes, 
                                 ensemble_weights_path, exp_name=None):
        """Initializes the voting layer for the ensemble based on the voting strategy."""
        if self.voting_strategy == 'weighted_model':
            layer = WeightedModelLayer(num_models)
        elif self.voting_strategy == 'weighted_families':
            layer = WeightedFamiliesLayer(num_models, num_classes)
        elif self.voting_strategy == 'weighted_families_pytorch':
            layer = WeightedFamiliesLayerPytorch(num_models, num_classes)
        else:
            return None, None

        # Define the file name based on the voting strategy and experiment name (if provided)
        if exp_name:
            file_name = f"{self.voting_strategy}_ensemble_{exp_name}.pt"
        else:
            file_name = f"{self.voting_strategy}_ensemble.pt"

        # If ensemble_weights_path is provided, use it to load weights
        if ensemble_weights_path:
            weights_file = f"{ensemble_weights_path}{file_name}"
        else:
            weights_file = f"{self.path}{file_name}"

        if ensemble_weights_path and os.path.exists(weights_file):
            try:
                layer.load_state_dict(tr.load(weights_file))
                print(f"Loaded voting layer weights from {weights_file}")
            except Exception:
                # Fallback for old format (raw tensor)
                weights = tr.load(weights_file)
                layer.weights.data.copy_(weights)
                print(f"Loaded raw weights into voting layer from {weights_file}")
        elif ensemble_weights_path:
            print(f"Warning: {weights_file} not found, using random init.")
            
        return layer, weights_file

    def _combine_ensemble_predictions(self, stacked_preds):
        """ Combines predictions from the ensemble models based on the voting strategy."""
        if self.voting_strategy == 'score_voting':
            pred = tr.mean(stacked_preds, dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy in ['weighted_model', 'weighted_families', 'weighted_mlp']:
            pred = self.voting_layer(stacked_preds) # forward pass through voting layer
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'simple_voting':
            pred_classes = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            pred = tr.nn.functional.one_hot(pred_classes, num_classes=len(self.categories)).float()
            pred_bin = tr.argmax(pred, dim=1)

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        return pred, pred_bin