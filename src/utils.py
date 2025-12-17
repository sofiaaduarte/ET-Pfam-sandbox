import os
import time
import psutil
import json
import threading
import pynvml as nvml
import pandas as pd
import numpy as np
import torch as tr
from torch.nn.functional import softmax
from contextlib import contextmanager

def predict(net, emb, window_len, use_softmax=True, step=8):
    """
    Predicts using a sliding window on the given embeddings.
    Args:
        net: BaseModel or Ensemble Model.
        emb: The input embeddings of shape (batch_size, sequence_length).
        window_len: The length of the sliding window.
        use_softmax: Whether to apply softmax to the predictions.
        step: Step size for the sliding window.
    Returns:
        centers: The center positions of the sliding windows.
        pred: The predictions from the model.
    """
    L = emb.shape[1]
    centers = np.arange(0, L, step)
    batch = tr.zeros((len(centers), emb.shape[0], window_len), dtype=tr.float)

    for k, center in enumerate(centers):
        start = max(0, center-window_len//2)
        end = min(L, center+window_len//2)
        batch[k,:,:end-start] = emb[:, start:end].unsqueeze(0)
    with tr.no_grad():
        pred = net(batch).cpu().detach()
    if use_softmax:
        pred = softmax(pred, dim=1)

    return centers, pred

def load_config(path='config/base.json'):
    """
    Loads a model configuration and merges it with environment-specific settings.
    Args:
        path (str): Path to the model config JSON file.
    Returns:
        dict: Combined configuration dictionary.
    """
    # Load model config from given path
    with open(path, 'r') as f:
        model = json.load(f)
    
    # Load env config from default path
    with open('config/env.json', 'r') as f:
        env = json.load(f)

    # Initialize config with model settings
    config = {**model}

    # Add environment-specific settings 
    keys_to_add = ['nworkers', 'device', 'emb_path', 'continue_training']
    for key in keys_to_add: 
        config[key] = env[key]
        
    # Add the path to the datase
    if config['dataset'] in ["full", "mini"]:
        config['data_path'] = env[f'{model["dataset"]}_path']
    else:
        raise ValueError(f"Invalid dataset name: {model['dataset']}. Expected 'full' or 'mini'.")

    return config

class ResultsTable():
    """Save results in a DataFrame and export to CSV."""
    
    def __init__(self, is_ensemble=False):
        """Initializes the ResultsTable"""
        self.is_ensemble = is_ensemble
        self.label_name = "Model" if not is_ensemble else "Strategy"
        self.df = pd.DataFrame(columns=[self.label_name, "CwS", "SwA", "SwC"])

    def add_entry(self, label, cws, swa, swc):
        """Add a new entry to the results DataFrame"""
        new_row = {
            self.label_name: label,
            "CwS": round(cws, 2),
            "SwA": round(swa, 2),
            "SwC": round(swc, 2)
        }
        self.df.loc[len(self.df)] = new_row

    def save(self, filepath):
        """Save the results DataFrame to a CSV file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.df.to_csv(filepath, index=False)

class ResourceMonitor:
    """Simple monitor to track time and memory usage."""
    
    def __init__(self, log_path):
        """
        Args:
            log_path (str): Path to the log file
        """
        self.log_path = log_path
        self.process = psutil.Process()
        
        # Initialize NVML for GPU monitoring
        self.nvml_handle = None
        if tr.cuda.is_available():
            nvml.nvmlInit()
            self.nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Header
        with open(self.log_path, 'w') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"=== Monitoring started: {timestamp} ===\n\n")
    
    @contextmanager
    def track(self, stage_name):
        """Context manager to track a stage of the process."""
        # Clear GPU cache if available
        gpu_available = tr.cuda.is_available()
        if gpu_available:
            tr.cuda.empty_cache()
            tr.cuda.reset_peak_memory_stats()
        
        # Variables for averages
        ram_samples = []
        gpu_allocated_samples = []
        gpu_reserved_samples = []
        gpu_nvml_samples = []
        monitoring = True
        
        def monitor_resources():
            """Function that runs in background taking samples."""
            while monitoring:
                ram_samples.append(self.process.memory_info().rss / 1024**3)
                if gpu_available:
                    gpu_allocated_samples.append(tr.cuda.memory_allocated() / 1024**3)
                    gpu_reserved_samples.append(tr.cuda.memory_reserved() / 1024**3)
                    # Get nvidia-smi value
                    if self.nvml_handle is not None:
                        info = nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                        gpu_nvml_samples.append(info.used / 1024**3)
                time.sleep(0.5)  # Sample every 0.5 seconds
        
        # Start
        start_time = time.time()
        self._write(f"[{time.strftime('%H:%M:%S')}] Starting: {stage_name}")
        
        # Start background monitoring
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        try:
            yield # Execute the block of code being monitored
        finally:
            # Stop monitoring
            monitoring = False
            monitor_thread.join(timeout=1.0)
            
            # End
            duration = time.time() - start_time
            end_ram = self.process.memory_info().rss / 1024**3
            
            # Calculate averages and peaks
            avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else end_ram
            peak_ram = max(ram_samples) if ram_samples else end_ram
            
            if gpu_available:
                peak_gpu_allocated = tr.cuda.max_memory_allocated() / 1024**3
                peak_gpu_reserved = tr.cuda.max_memory_reserved() / 1024**3
                avg_gpu_allocated = sum(gpu_allocated_samples) / len(gpu_allocated_samples) if gpu_allocated_samples else 0
                avg_gpu_reserved = sum(gpu_reserved_samples) / len(gpu_reserved_samples) if gpu_reserved_samples else 0
                
                # NVML (nvidia-smi) stats if available
                peak_gpu_nvml = max(gpu_nvml_samples) if gpu_nvml_samples else 0
                avg_gpu_nvml = sum(gpu_nvml_samples) / len(gpu_nvml_samples) if gpu_nvml_samples else 0
            
            self._write(f"[{time.strftime('%H:%M:%S')}] Finished: {stage_name}")
            self._write(f"  Time: {duration/60:.2f} min ({duration:.1f} sec)")
            self._write(f"  RAM: {peak_ram:.2f} GB peak (avg: {avg_ram:.2f} GB)")
            if gpu_available:
                self._write(f"  GPU Allocated: {peak_gpu_allocated:.2f} GB peak (avg: {avg_gpu_allocated:.2f} GB)")
                self._write(f"  GPU Reserved:  {peak_gpu_reserved:.2f} GB peak (avg: {avg_gpu_reserved:.2f} GB)")
                if self.nvml_handle is not None and peak_gpu_nvml > 0:
                    self._write(f"  GPU Total (nvidia-smi): {peak_gpu_nvml:.2f} GB peak (avg: {avg_gpu_nvml:.2f} GB)")
            self._write("")
    
    def _write(self, message):
        """Write to log and force flush."""
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")
            f.flush() # Ensure it's written to disk immediately
    
    def log(self, message):
        """Write a custom message."""
        self._write(f"[{time.strftime('%H:%M:%S')}] {message}")

class DummyMonitor:
    """Dummy monitor that does nothing when monitor is None."""
    
    @contextmanager
    def track(self, stage_name):
        """No-op context manager."""
        yield
    
    def log(self, message):
        """No-op log."""
        pass
