"""
Utilities for analyzing attention patterns in transformer models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import einops

def analyze_object_type_attention(model, cache, object_types, padding_token, 
                                 ret_dict=False, combine_elec_and_muon=False, 
                                 exclude_self=False):
    """
    Analyze how different object types attend to each other.
    
    Args:
        model: The neural network model
        cache: ActivationCache with model activations
        object_types: Tensor of object types [batch, object]
        padding_token: Token ID representing padding
        ret_dict: Whether to return results as dict instead of array
        combine_elec_and_muon: Whether to combine electron and muon types
        exclude_self: Whether to exclude self-attention
        
    Returns:
        Analysis of attention patterns by object type
    """
    object_types = object_types.clone()
    if combine_elec_and_muon:
        object_types[object_types==0] = 1
        object_types -= 1
        padding_token -= 1
    results = {}
    
    for block_idx in range(model.num_attention_blocks):
        attn_weights = cache[f"block_{block_idx}_attention"]["attn_weights_per_head"]
        
        # For each head, analyze attention patterns between object types
        num_heads = attn_weights.shape[1]
        block_results = {}
        
        for head_idx in range(num_heads):
            head_attn = attn_weights[:, head_idx]  # [batch, query, key]
            
            # Average attention by object type pairs
            unique_types = torch.unique(object_types)
            if ret_dict:
                type_attention = {}
            else:
                type_attention = np.zeros((len(unique_types), len(unique_types)))
            
            for query_type in unique_types:
                for key_type in unique_types:
                    # Skip padding tokens
                    if query_type == (padding_token) or key_type == (padding_token):
                        continue
                        
                    # Create masks for the query and key object types
                    query_mask = (object_types == query_type).unsqueeze(-1)  # [batch, query, 1]
                    key_mask = (object_types == key_type).unsqueeze(1)  # [batch, 1, key]
                    
                    if exclude_self:
                        not_diag_mask = (~(torch.eye(object_types.shape[-1]).to(bool))).unsqueeze(0) # [1 query key]
                    else:
                        not_diag_mask = torch.ones_like(key_mask) # [batch, 1, key]

                    # Apply masks and compute average attention
                    masked_attn = head_attn * query_mask * key_mask * not_diag_mask
                    
                    # Sum over the keys to get the total attention paid TO this type of object
                    # Then normalize by how many of these objects appear in TOTAL.
                    q_entries = einops.einsum(query_mask, 'batch query key ->')
                    if q_entries > 0:
                        avg_attn = einops.einsum(masked_attn, 'batch query key ->') / q_entries.item()
                        if ret_dict:
                            type_attention[f"{query_type.item()}->{key_type.item()}"] = avg_attn.item()
                        else:
                            type_attention[query_type.item(), key_type.item()] = avg_attn.item()
                    else:
                        if ret_dict:
                            type_attention[f"{query_type.item()}->{key_type.item()}"] = 0
                        else:
                            type_attention[query_type.item(), key_type.item()] = 0
            
            block_results[f"head_{head_idx}"] = type_attention
        
        results[f"block_{block_idx}"] = block_results
    
    return results

class AttentionAnalyzer:
    """Analyze attention patterns in transformer models."""
    
    def __init__(self, model, cache=None):
        """
        Args:
            model: The neural network model
            cache: Optional ActivationCache with model activations
        """
        self.model = model
        self.cache = cache
        
    def set_cache(self, cache):
        """Set the activation cache for analysis."""
        self.cache = cache
        
    def analyze_type_attention(self, object_types, padding_token, 
                              combine_elec_and_muon=False, 
                              exclude_self=False):
        """Analyze attention patterns between object types.
        
        Args:
            object_types: Tensor of object types
            padding_token: Token ID for padding
            combine_elec_and_muon: Whether to combine electron and muon types
            exclude_self: Whether to exclude self-attention
            
        Returns:
            Dictionary of attention patterns by layer and head
        """
        return analyze_object_type_attention(
            self.model, 
            self.cache, 
            object_types, 
            padding_token,
            combine_elec_and_muon=combine_elec_and_muon,
            exclude_self=exclude_self
        )
        
    def visualize_type_attention(self, attention_patterns, type_names, 
                                layer_range=None, head_range=None, 
                                figsize=None):
        """Visualize attention patterns between object types.
        
        Args:
            attention_patterns: Results from analyze_type_attention
            type_names: Dictionary mapping type IDs to names
            layer_range: Optional range of layers to visualize
            head_range: Optional range of heads to visualize
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        if layer_range is None:
            layer_range = range(self.model.num_attention_blocks)
        if head_range is None:
            head_range = range(self.model.attention_blocks[0]['self_attention'].num_heads)
        
        if figsize is None:
            figsize = (len(head_range)*3, len(layer_range)*3)
        fig, axes = plt.subplots(
            len(layer_range), 
            len(head_range), 
            figsize=figsize, 
            squeeze=False
        )
        
        for i, layer_idx in enumerate(layer_range):
            for j, head_idx in enumerate(head_range):
                ax = axes[i, j]
                data = attention_patterns[f"block_{layer_idx}"][f"head_{head_idx}"]
                
                # Convert to numpy array if it's a dictionary
                if isinstance(data, dict):
                    # Create matrix from dictionary
                    types = sorted(list(set([int(k.split('->')[0]) for k in data.keys()])))
                    matrix = np.zeros((len(types), len(types)))
                    for k, v in data.items():
                        src, dst = map(int, k.split('->'))
                        src_idx = types.index(src)
                        dst_idx = types.index(dst)
                        matrix[src_idx, dst_idx] = v
                    data = matrix
                
                # Plot heatmap
                sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis', ax=ax, vmin=0, vmax=1,
                           xticklabels=[type_names.get(t, f"Type {t}") for t in range(data.shape[1])],
                           yticklabels=[type_names.get(t, f"Type {t}") for t in range(data.shape[0])])
                ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
                ax.set_xlabel("Key Type")
                ax.set_ylabel("Query Type")
                
        plt.tight_layout()
        return fig 
    