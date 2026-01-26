This branch contains the novel proposed ensemble strategies. To use this branch:

```bash
git clone https://github.com/sofiaaduarte/ET-Pfam-sandbox.git
cd ET-Pfam-sandbox
git checkout ensemble_strategy
```

This branch includes new ensemble strategies using PyTorch `Linear` layers:
- `family_linear`: *Learned weights by family (LWF) perceptron voting*. This is the same as the original LWF but implemented with a PyTorch `Linear` layer.
- `family_mlp_linear`: *Learned weights by family MLP voting*
- `flatten_linear`: *Learned stacking perceptron voting*
- `flatten_mlp`: *Learned stacking MLP voting*

These strategies can be trained and tested individually using the unified `train_test_ensemble.py` script.

## Configure and run a single experiment

Create or modify `config/ensemble.json` with your desired configuration:

```json
{
    "voting_strategy": "flatten_mlp",
    "use_bias": true,
    "hidden_size": 128,
    "learning_rate": 0.001,
    "n_epochs": 500
}
```

Then run the training and testing, for the mini dataset, with:

```bash
python3 train_test_ensemble.py -m models/mini/
```