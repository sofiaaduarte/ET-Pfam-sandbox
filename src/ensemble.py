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

class ModelWeights(nn.Module):
    """Simple ensemble, with one weight per model. 
    This is the original weighted model layer."""
    def __init__(self, num_models):
        super(ModelWeights, self).__init__()
        self.weights = nn.Parameter(tr.rand(num_models))

    def forward(self, x): # x: (num_models, batch, num_classes)
        # reshape for broadcasting (num_models, 1, 1)
        w = self.weights.view(-1, 1, 1) 

        # weighted sum over models - output shape: (batch, num_classes)
        return tr.sum(x * w, dim=0) 

class FamilyWeights(nn.Module):
    """Weighted family ensemble, with one weight per model per family.
    This is the original weighted family layer."""
    def __init__(self, num_models, num_classes, bias=False):
        super(FamilyWeights, self).__init__()
        self.weights = nn.Parameter(tr.rand(num_models, num_classes))
        if bias:
            self.bias = nn.Parameter(tr.zeros(num_classes))
        else:
            self.bias = None

    def forward(self, x): # x: (num_models, batch, num_classes)
        # reshape for broadcasting (num_models, 1, num_classes)
        w = self.weights.view(self.weights.shape[0], 1, -1) 

        # weighted sum over models
        out = tr.sum(x * w, dim=0)
        if self.bias is not None:
            out = out + self.bias

        return out # (batch, num_classes)

class FamilyMLP(nn.Module):
    """Two-layer MLP ensemble."""
    def __init__(self, num_models, num_classes, hidden_size=4):
        super().__init__()

        # First layer (one MLP per class)
        self.weights1 = nn.Parameter(tr.rand(num_models, num_classes, hidden_size))
        self.bias1 = nn.Parameter(tr.zeros(num_classes, hidden_size))

        # Second layer (one MLP per class)
        self.weights2 = nn.Parameter(tr.rand(num_classes, hidden_size))
        self.bias2 = nn.Parameter(tr.zeros(num_classes))

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Expand x to (num_models, batch, num_classes, 1)
        x = x.unsqueeze(-1)

        # First layer
        w1 = self.weights1.unsqueeze(1) # (num_models, 1, num_classes, hidden_size)
        b1 = self.bias1.unsqueeze(0) # (1, num_classes, hidden_size)
        h = tr.sum(x * w1, dim=0) + b1
        h = tr.relu(h)  # (batch, num_classes, hidden_size)

        # Second layer
        w2 = self.weights2.unsqueeze(0) # (1, num_classes, hidden_size)
        b2 = self.bias2 # (num_classes,)
        out = tr.sum(h * w2, dim=-1) + b2  # (batch, num_classes)

        return out

class FamilyLinear(nn.Module):
    """Original weighted family layer, but refactored using PyTorch Linear layers"""
    def __init__(self, num_models, num_classes, bias=False):
        super(FamilyLinear, self).__init__()
        # Create a Linear layer for each class (the inputs are the model scores, 
        # the output is the weighted final score)
        self.linears = nn.ModuleList([
            nn.Linear(num_models, 1, bias=bias) for _ in range(num_classes)
        ])

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Transpose to (batch, num_classes, num_models), because each linear 
        # layer expects input of shape (batch, num_models)
        x = x.permute(1, 2, 0)
        
        # Apply each linear layer to its corresponding class
        outputs = []
        for i, linear in enumerate(self.linears):
            # select scores for class i: (batch, num_models)
            x_class = x[:, i, :]
            class_score = linear(x_class)  # (batch, 1)
            outputs.append(class_score)
        
        # Concatenate to get (batch, num_classes)
        out = tr.cat(outputs, dim=1)

        return out  # (batch, num_classes)
    
class FamilyMLPLinear(nn.Module):
    """Two-layer MLP ensemble using PyTorch Linear layers."""
    def __init__(self, num_models, num_classes, hidden_size=4):
        super(FamilyMLPLinear, self).__init__()

        # First layer (one MLP per class)
        self.linears1 = nn.ModuleList([
            nn.Linear(num_models, hidden_size) for _ in range(num_classes)
        ])

        # Second layer
        self.linears2 = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_classes)
        ])

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Transpose to (batch, num_classes, num_models)
        x = x.permute(1, 2, 0)

        # First layer
        h_list = []
        for i, linear1 in enumerate(self.linears1):
            x_class = x[:, i, :]  # (batch, num_models)
            h = linear1(x_class)  # (batch, hidden_size)
            h = tr.relu(h)
            h_list.append(h)
        h = tr.stack(h_list, dim=1)  # (batch, num_classes, hidden_size)

        # Second layer
        out_list = []
        for i, linear2 in enumerate(self.linears2):
            h_class = h[:, i, :]  # (batch, hidden_size)
            out_class = linear2(h_class)  # (batch, 1)
            out_list.append(out_class)
        out = tr.cat(out_list, dim=1)  # (batch, num_classes)

        return out

class FlattenLinear(nn.Module):
    """Flatten all predictions into a single vector and apply one Linear layer."""
    def __init__(self, num_models, num_classes, bias=True):
        super(FlattenLinear, self).__init__()
        self.linear = nn.Linear(num_models * num_classes, num_classes, bias=bias)

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Permute to (batch, num_models, num_classes)
        x = x.permute(1, 0, 2)
        
        # Flatten to (batch, num_models * num_classes)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Apply linear layer
        out = self.linear(x_flat)

        return out  # (batch, num_classes)
    
class FlattenMLP(nn.Module):
    """Flatten all predictions and apply a two-layer MLP."""
    def __init__(self, num_models, num_classes, hidden_size=64, bias=True):
        super(FlattenMLP, self).__init__()
        input_size = num_models * num_classes
        
        # Two-layer MLP
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Permute to (batch, num_models, num_classes)
        x = x.permute(1, 0, 2)
        
        # Flatten to (batch, num_models * num_classes)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # First layer with ReLU activation
        h = self.fc1(x_flat)  # (batch, hidden_size)
        h = tr.relu(h)
        
        # Second layer to output
        out = self.fc2(h)

        return out  # (batch, num_classes)
    
class EnsembleModel(nn.Module):
    def __init__(self, models_path, config, voting_strategy, 
                 ensemble_weights_path=None, exp_name=None, hidden_size=4,
                 use_bias=True):
        super(EnsembleModel, self).__init__()

        # Load model paths
        model_dirs = [os.path.join(models_path, d) for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        
        self.emb_path = config['emb_path']
        self.data_path = config['data_path']
        self.voting_strategy = voting_strategy
        self.path = models_path
        self.weights_file = None
        self.use_bias = use_bias

        self.available_weighted_strategies = {
            'weighted_model': ModelWeights,
            'weighted_families': FamilyWeights,
            'weighted_families_mlp': FamilyMLP,
            'family_linear': FamilyLinear,
            'family_mlp_linear': FamilyMLPLinear,
            'flatten_linear': FlattenLinear,
            'flatten_mlp': FlattenMLP
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
            len(model_dirs), 
            len(self.categories), 
            ensemble_weights_path, 
            exp_name,
            hidden_size=hidden_size,
            use_bias=self.use_bias
        )

    def fit(self, learning_rate=0.01, n_epochs=500):
        if self.voting_strategy in self.available_weighted_strategies:
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
                dev_loader = tr.utils.data.DataLoader(
                    dev_data, 
                    batch_size=config['batch_size'], 
                    num_workers=config.get("nworkers", 1)
                    )

                with tr.no_grad():
                    _, _, pred, ref, *_ = net.pred(dev_loader)
                    all_preds.append(pred)
            stacked_preds = tr.stack(all_preds)

            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam(self.voting_layer.parameters(), lr=learning_rate)

            for epoch in tqdm(range(n_epochs), desc="Epochs"):
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
            test_loader = tr.utils.data.DataLoader(
                test_data, 
                batch_size=config['batch_size'], 
                num_workers=config.get("nworkers", 1)
                )
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
                                 ensemble_weights_path, exp_name=None, 
                                 hidden_size=4, use_bias=True):
        """Initializes the voting layer for the ensemble based on the voting strategy."""
        # Define the voting layer based on the strategy

        # Original weighted strategies, and a new one using bias and MLP
        if self.voting_strategy == 'weighted_model':
            layer = ModelWeights(num_models)
        elif self.voting_strategy == 'weighted_families':
            layer = FamilyWeights(num_models, num_classes, bias=use_bias)
        elif self.voting_strategy == 'weighted_families_mlp':
            layer = FamilyMLP(num_models, num_classes, hidden_size=hidden_size) # always with bias
        # Orignal strategies refactored using Linear layers
        elif self.voting_strategy == 'family_linear':
            layer = FamilyLinear(num_models, num_classes, bias=use_bias)
        elif self.voting_strategy == 'family_mlp_linear':
            layer = FamilyMLPLinear(num_models, num_classes, hidden_size=hidden_size) # always with bias
        # New strategies using flattening
        elif self.voting_strategy == 'flatten_linear':
            layer = FlattenLinear(num_models, num_classes, bias=use_bias)
        elif self.voting_strategy == 'flatten_mlp':
            layer = FlattenMLP(num_models, num_classes, hidden_size=hidden_size, bias=use_bias)
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

        elif self.voting_strategy in self.available_weighted_strategies:
            pred = self.voting_layer(stacked_preds) # forward pass through voting layer
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'simple_voting':
            pred_classes = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            pred = tr.nn.functional.one_hot(pred_classes, num_classes=len(self.categories)).float()
            pred_bin = tr.argmax(pred, dim=1)

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        return pred, pred_bin