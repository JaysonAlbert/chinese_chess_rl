import torch
import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import logging

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))
from xiangqi_rl.model import XiangqiHybridNet

def setup_logging(save_dir):
    """Setup logging to both file and console"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'weight_analysis_{timestamp}.log')
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('weight_analysis')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_checkpoint(path):
    """Load a checkpoint and return the model state dict"""
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint

def analyze_weights(state_dict):
    """Analyze weights of the model and return statistics"""
    stats = defaultdict(dict)
    
    for name, param in state_dict.items():
        if 'weight' in name:
            weights = param.cpu().numpy()
            stats[name] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'sparsity': float(np.mean(np.abs(weights) < 1e-6)),
                'magnitude': float(np.mean(np.abs(weights))),
                'shape': weights.shape
            }
            
            # Additional conv layer specific analysis
            if 'conv' in name.lower():
                # Analyze filter diversity
                filters = weights.reshape(weights.shape[0], -1)
                correlation = np.corrcoef(filters)
                stats[name]['filter_correlation'] = float(np.mean(np.abs(correlation - np.eye(correlation.shape[0]))))
                
                # Analyze kernel structure
                if len(weights.shape) == 4:
                    stats[name]['kernel_structure'] = {
                        'center_mass': float(np.mean(np.abs(weights[:, :, 1:4, 1:4]))) / 
                                     float(np.mean(np.abs(weights)))
                    }
    
    return stats

def compare_checkpoints(checkpoint_paths, logger):
    """Compare weights between different checkpoints"""
    all_stats = {}
    
    for path in checkpoint_paths:
        logger.info(f"\nAnalyzing checkpoint: {path}")
        state_dict = load_checkpoint(path)
        stats = analyze_weights(state_dict)
        all_stats[path] = stats
        
        # Print summary for each layer
        for name, layer_stats in stats.items():
            logger.info(f"\nLayer: {name}")
            logger.info(f"Shape: {layer_stats['shape']}")
            logger.info(f"Mean: {layer_stats['mean']:.6f}")
            logger.info(f"Std: {layer_stats['std']:.6f}")
            logger.info(f"Magnitude: {layer_stats['magnitude']:.6f}")
            logger.info(f"Sparsity: {layer_stats['sparsity']:.2%}")
            
            if 'filter_correlation' in layer_stats:
                logger.info(f"Filter correlation: {layer_stats['filter_correlation']:.4f}")
            if 'kernel_structure' in layer_stats:
                logger.info(f"Center/Overall ratio: {layer_stats['kernel_structure']['center_mass']:.4f}")
    
    return all_stats

def plot_weight_distributions(all_stats, save_dir):
    """Plot weight distribution changes across checkpoints"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all layer names
    layer_names = list(next(iter(all_stats.values())).keys())
    checkpoints = list(all_stats.keys())
    
    # Extract iteration numbers for x-axis labels
    iterations = []
    for ckpt in checkpoints:
        try:
            iter_num = int(Path(ckpt).stem.split('_')[-1])
            iterations.append(iter_num)
        except:
            iterations.append(len(iterations))
    
    # Plot evolution of key metrics for each conv layer
    metrics = ['magnitude', 'std', 'filter_correlation']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for name in layer_names:
            if 'conv' in name.lower():
                values = [all_stats[ckpt][name].get(metric, 0) for ckpt in checkpoints]
                plt.plot(iterations, values, label=name, marker='o')
        
        plt.title(f'Evolution of {metric}')
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_evolution.png'))
        plt.close()

def main():
    # Setup save directory
    save_dir = Path(__file__).parent.parent / "logs" / "weight_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir)
    
    # Get checkpoints from command line arguments or use default pattern
    if len(sys.argv) > 1:
        checkpoint_paths = sys.argv[1:]
    else:
        # Default pattern for checkpoints
        checkpoint_dir = Path(__file__).parent.parent / "logs" / "checkpoints"
        checkpoint_paths = sorted(
            checkpoint_dir.glob("model_iteration_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )
    
    logger.info(f"Analyzing checkpoints: {checkpoint_paths}")
    
    # Compare checkpoints
    all_stats = compare_checkpoints(checkpoint_paths, logger)
    
    # Plot distributions
    plot_weight_distributions(all_stats, save_dir)
    
    # Analyze potential issues
    logger.info("\nPotential issues detected:")
    for ckpt_path, stats in all_stats.items():
        logger.info(f"\nCheckpoint: {ckpt_path}")
        
        for name, layer_stats in stats.items():
            if 'conv' in name.lower():
                # Check for potential issues that might slow down convolutions
                if layer_stats['std'] > 2.0:
                    logger.warning(f"- High variance in {name}: std={layer_stats['std']:.4f}")
                if layer_stats['filter_correlation'] > 0.5:
                    logger.warning(f"- High filter correlation in {name}: {layer_stats['filter_correlation']:.4f}")
                if layer_stats['sparsity'] > 0.5:
                    logger.warning(f"- High sparsity in {name}: {layer_stats['sparsity']:.2%}")
                if layer_stats.get('kernel_structure', {}).get('center_mass', 0) < 0.2:
                    logger.warning(f"- Unusual kernel structure in {name}")

if __name__ == "__main__":
    main() 