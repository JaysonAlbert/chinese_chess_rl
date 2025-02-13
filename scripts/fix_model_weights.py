import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from xiangqi_rl.model import XiangqiHybridNet

def fix_weights(state_dict):
    """Fix problematic weight patterns"""
    fixed_dict = {}
    for name, param in state_dict.items():
        if 'conv' in name and 'weight' in name:
            # Get weight tensor
            weights = param.clone()
            
            # 1. Fix extreme sparsity by pruning and rescaling
            mask = torch.abs(weights) > weights.abs().mean() * 0.1
            weights = weights * mask
            
            # 2. Normalize remaining weights to maintain activation scale
            if mask.sum() > 0:
                scale = param.abs().mean() / weights.abs().mean()
                weights = weights * scale
            
            # 3. Add small noise to break filter correlation
            if 'conv2' in name:
                noise = torch.randn_like(weights) * weights.std() * 0.1
                weights = weights + noise
                
            fixed_dict[name] = weights
        else:
            fixed_dict[name] = param.clone()
    
    return fixed_dict

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_model_weights.py input_checkpoint output_checkpoint")
        return
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load checkpoint
    checkpoint = torch.load(input_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Fix weights
    fixed_dict = fix_weights(state_dict)
    
    # Save fixed checkpoint
    if isinstance(checkpoint, dict):
        checkpoint['model_state_dict'] = fixed_dict
        torch.save(checkpoint, output_path)
    else:
        torch.save(fixed_dict, output_path)
    
    print(f"Fixed weights saved to {output_path}")

if __name__ == "__main__":
    main() 