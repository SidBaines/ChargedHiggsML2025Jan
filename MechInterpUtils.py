import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
from utils import Get_PtEtaPhiM_fromXYZT, GetXYZT_FromPtEtaPhiM
import einops
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import math

ATTN_METHOD_1 = False

class ActivationCache:
    """Store activations from model hooks."""
    def __init__(self):
        self.store = defaultdict(dict)
        
    def __getitem__(self, key):
        return self.store[key]
    
    def clear(self):
        self.store.clear()

def run_with_hooks(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, torch.Tensor],
    fwd_hooks: List[Tuple[nn.Module, Callable]] = None,
    bwd_hooks: List[Tuple[nn.Module, Callable]] = None,
    clear_hooks: bool = True
) -> torch.Tensor:
    """
    Run model with temporary hooks attached.
    
    Args:
        model: The neural network model
        inputs: Tuple of (object_features, object_types)
        fwd_hooks: List of (module, hook_fn) pairs for forward hooks
        bwd_hooks: List of (module, hook_fn) pairs for backward hooks
        clear_hooks: Whether to remove hooks after running
        
    Returns:
        Model output
    """
    hooks = []
    
    try:
        # Register forward hooks
        if fwd_hooks:
            for module, hook_fn in fwd_hooks:
                hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
        
        # Register backward hooks
        if bwd_hooks:
            for module, hook_fn in bwd_hooks:
                hooks.append(module.register_backward_hook(hook_fn, with_kwargs=True))
        
        # Run the model
        output = model(*inputs)
        
        return output
    
    finally:
        # Clean up by removing hooks
        if clear_hooks:
            for hook in hooks:
                hook.remove()

def get_activation_hook(cache: ActivationCache, name: str, detach=True):
    """Create a hook function that saves activations to the cache."""
    if detach:
        def hook_fn(module, input, kwargs, output):
            cache.store[name]["input"] = [x.detach() if isinstance(x, torch.Tensor) else x for x in input]
            cache.store[name]["output"] = output.detach()
    else:
        # print("WARNING: Running cache with non-detached output - not sure how safe this is...")
        def hook_fn(module, input, kwargs, output):
            cache.store[name]["input"] = [x if isinstance(x, torch.Tensor) else x for x in input]
            cache.store[name]["output"] = output
    return hook_fn

def get_gradient_hook(cache: ActivationCache, name: str, detach=True):
    """Create a hook function that saves gradients to the cache."""
    if detach:
        def hook_fn(module, grad_input, kwargs, grad_output):
            cache.store[name]["grad_input"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_input]
            cache.store[name]["grad_output"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_output]
    else:
        # print("WARNING: Running cache with non-detached output - not sure how safe this is...")
        def hook_fn(module, grad_input, kwargs, grad_output):
            cache.store[name]["grad_input"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_input]
            cache.store[name]["grad_output"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_output]
    return hook_fn

def hook_attention_heads(model, cache: ActivationCache, detach=True):
    """Add hooks to extract attention patterns and outputs from each head."""
    hooks = []
    
    # Hook attention blocks
    for i, block in enumerate(model.attention_blocks):
        # Original MultiheadAttention's forward is complex, so we'll need to hook it specially
        def get_attention_hook(block_idx, detach=True):  
            def hook_fn(module, inputs, kwargs, output):
                # Extract q, k, v and attention weights (output is just attn_output)
                # For nn.MultiheadAttention, output is (attn_output, attn_weights)
                attn_output, attn_weights = output
                if detach:
                    cache.store[f"block_{block_idx}_attention"]["input"] = [x.detach() if isinstance(x, torch.Tensor) else x for x in inputs]
                    cache.store[f"block_{block_idx}_attention"]["attn_weights"] = attn_weights.detach() # Can keep these as they are actually fine
                    cache.store[f"block_{block_idx}_attention"]["OLD_attn_output"] = attn_output.detach() # This is averaged over head, so we shouldn't use it since we now have per-head
                else:
                    cache.store[f"block_{block_idx}_attention"]["input"] = [x if isinstance(x, torch.Tensor) else x for x in inputs]
                    cache.store[f"block_{block_idx}_attention"]["attn_weights"] = attn_weights # Can keep these as they are actually fine
                    cache.store[f"block_{block_idx}_attention"]["OLD_attn_output"] = attn_output # This is averaged over head, so we shouldn't use it since we now have per-head
                key_padding_mask = kwargs.get('key_padding_mask', None)
                key_padding_mask = torch.nn.functional._canonical_mask(
                    mask=key_padding_mask,
                    mask_name="key_padding_mask",
                    other_type=torch.nn.functional._none_or_dtype(key_padding_mask),
                    other_name="UNSURE",
                    target_type=inputs[0].dtype,
                )
                q, k, v = inputs
                B, N, E = v.shape
                # q = q.view(B, N, module.num_heads, E // module.num_heads)
                # k = k.view(B, N, module.num_heads, E // module.num_heads)
                # v = v.view(B, N, module.num_heads, E // module.num_heads)
                # TODO are we *sure* we don't need to project on by module.v_proj_weight or something here? tbf it doesn't seem to exist so that's some indication that we don't...
                # v = module.v_proj_weight ... v
                if not ((q==k).all() and (q==v).all()): # Need to make sure this is self attention; haven't set up for anything else:
                    raise NotImplementedError
                else:
                    q = k = v = q.transpose(1,0) # After this line, shape is now: [object batch dmodel]
                    tgt_len, bsz, embed_dim  = q.shape
                    src_len, _, _  = k.shape
                    # Literally copied from here for a quick hack/lack of a better way: https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L6230
                    q, k, v = torch.nn.functional._in_projection_packed(q, k, v, module.in_proj_weight, module.in_proj_bias) 
                    # Reshape for multihead and with batch first
                    num_heads = module.num_heads
                    head_dim = E // num_heads
                    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # After this line, shape is now: [batch*head object_query dhead]
                    k = k.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # After this line, shape is now: [batch*head object_key dhead]
                    v = v.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # After this line, shape is now: [batch*head object_value dhead]
                    _B, _Nt, E = q.shape
                    q_scaled = q * math.sqrt(1.0 / float(E))
                    # Now do what they do in (sorry if it's bad form to copy chunks of code like this, just need a quick hack!) https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L6368-L6378
                    if key_padding_mask is not None:
                        key_padding_mask = (
                            key_padding_mask.view(bsz, 1, 1, src_len)
                            .expand(-1, num_heads, -1, -1)
                            .reshape(bsz * num_heads, 1, src_len)
                        )   # After this line, shape is now: [batch*head 1 object]
                        attn_weights_byhand = torch.baddbmm(
                            key_padding_mask, q_scaled, k.transpose(-2, -1)
                        ) # After this line, shape is now: [batch*head object_query object_key]
                    else:
                        attn_weights_byhand = torch.bmm(q_scaled, k.transpose(-2, -1))
                    attn_weights_byhand = torch.nn.functional.softmax(attn_weights_byhand, dim=-1) # After this line, shape is now: [batch*head object_query object_key]
                    if 1: # Little check
                        assert(torch.isclose(attn_weights_byhand.view(bsz, num_heads, tgt_len, src_len)[0,1,:,:], attn_weights[0,1,:,:], atol=1e-05).all()) # Check batch element 0, head 1, all queries/keys to make sure they match

                    attn_output_byhand = torch.bmm(attn_weights_byhand, v) # After this line, shape is now: [batch*head object_query dhead]
                    attn_output_byhand = (
                        attn_output_byhand.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
                    ) # After this line, shape is now: [object_query*batch headnum*dhead]
                    if 0: # Unnecessary, just for a check
                        attn_output_byhand_headcombined = torch.matmul(attn_output_byhand, module.out_proj.weight.transpose(0,1)) + module.out_proj.bias # After this line, shape is now: [object_query*batch dmodel] # NOTE that in multiplying by the outprojection, we have combined the heads in an inrreversible way
                        attn_output_byhand_headcombined = attn_output_byhand_headcombined.contiguous().view(tgt_len, bsz, -1).transpose(0,1) # After this line, shape is now: [batch object_query dmodel]
                    attn_output_per_head = torch.empty((B, module.num_heads, N, module.out_proj.weight.shape[0]))
                    for head_n in range(num_heads):
                        W_rows = module.out_proj.weight.transpose(0,1)[head_n*head_dim:(head_n+1)*head_dim] # After this line, shape is now [d_head dmodel]
                        attnoutput_cols = attn_output_byhand[:, head_n*head_dim:(head_n+1)*head_dim] # After this line, shape is now [object_query*batch d_head]
                        head_output = torch.matmul(attnoutput_cols, W_rows) # After this line, shape is now: [object_query*batch dmodel] # NOTE that since we took the relevant slice of the W matrix rows and relevant slice of attn_output columns for this head, we have only projected this head onto dmodel
                        attn_output_per_head[:, head_n, :, :] = head_output.contiguous().view(tgt_len, bsz, -1).transpose(0,1) # After this line, shape is now: [batch object_query dmodel]
                    if 1: # Just a quick test to check we got the output correctly
                        recrafted_attn_output=(attn_output_per_head.sum(dim=1) + module.out_proj.bias)
                        assert(torch.isclose(recrafted_attn_output[0,:5,0],attn_output[0,:5,0], atol=1e-05).all())
                        # print(torch.isclose(b,attn_output).sum()/b.numel()) # 0.9934 of the elements pass isclose
                if detach:
                    cache.store[f"block_{block_idx}_attention"]["attn_weights_per_head"] = attn_weights_byhand.view(bsz, num_heads, tgt_len, src_len).detach()
                    cache.store[f"block_{block_idx}_attention"]["attn_unprojected_output_per_head"] = attn_output_byhand.view(tgt_len, bsz, num_heads, head_dim).transpose(0,1).detach()
                    cache.store[f"block_{block_idx}_attention"]["attn_output_per_head"] = attn_output_per_head.detach()
                else:
                    # print("WARNING: Running cache with non-detached output - not sure how safe this is...")
                    cache.store[f"block_{block_idx}_attention"]["attn_weights_per_head"] = attn_weights_byhand.view(bsz, num_heads, tgt_len, src_len)
                    cache.store[f"block_{block_idx}_attention"]["attn_unprojected_output_per_head"] = attn_output_byhand.view(tgt_len, bsz, num_heads, head_dim).transpose(0,1)
                    cache.store[f"block_{block_idx}_attention"]["attn_output_per_head"] = attn_output_per_head
                return output
            return hook_fn


        def patch_attention(m):
            forward_orig = m.forward
            def wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            m.forward = wrap
        if str(type(block['self_attention'].forward)) == "<class 'method'>":
            # hasn't been patched yet
            patch_attention(block['self_attention'])
        elif str(type(block['self_attention'].forward)) == "<class 'function'>":
            # Has been patched already
            pass
        else:
            assert(False) # Shouldn't get here

        hooks.append((block['self_attention'], get_attention_hook(i, detach=detach)))


        
        # Hook the layernorm and post-attention module
        # hooks.append((block['layer_norm'], get_activation_hook(cache, f"block_{i}_layernorm")))
        # try:
        if model.include_mlp:
            hooks.append((block['post_attention'], get_activation_hook(cache, f"block_{i}_post_attention", detach=detach)))
        # except:
        else:
            # print("Tried to attach post-attention block but couldn't; this could be because your model doesn't have one (which is fine) or it could be something else (probably not fine)")
            pass
    
    # Hook the final classifier
    hooks.append((model.classifier, get_activation_hook(cache, "classifier", detach=detach)))
    
    return hooks

def extract_all_activations(model, object_features, object_types, detach=True):
    """
    Extract and return all important intermediate activations from the model.
    """
    cache = ActivationCache()
    
    # Create hooks for all components
    hooks = [
        (model.object_net, get_activation_hook(cache, "object_net", detach=detach)),
        (model.type_embedding, get_activation_hook(cache, "type_embedding", detach=detach))
    ]
    
    # Add hooks for attention blocks
    hooks.extend(hook_attention_heads(model, cache, detach=detach))
    
    # Run the model with hooks
    output = run_with_hooks(
        model,
        (object_features, object_types),
        fwd_hooks=hooks
    )
    
    # Add the output to the cache
    if detach:
        cache.store["output"] = output.detach()
    else:
        # print("WARNING: Running cache with non-detached output - not sure how safe this is...")
        cache.store["output"] = output
    
    return cache

def get_residual_stream(cache: ActivationCache):
    """
    Extract the residual stream from each attention block.
    
    Returns:
        Dict mapping block index to residual stream tensor
    """
    residuals = {}
    
    # First residual is the output of the object_net
    residuals[0] = cache["object_net"]["output"]
    
    # Extract residuals from each attention block
    for i in range(len([i for i in cache.store.keys() if 'post_attention' in i])):
        if f"block_{i}_post_attention" in cache.store:
            residuals[i+1] = cache[f"block_{i}_post_attention"]["input"][0] + cache[f"block_{i}_post_attention"]["output"] # The real residual stream is the input + the output to the post-attention block
            # residuals[i+1] = cache[f"block_{i}_post_attention"]["output"]
    
    return residuals






import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import seaborn as sns
from matplotlib.patches import Rectangle

def old_direct_logit_attribution(model, cache, class_idx=None, top_k=5):
    """
    Perform direct logit attribution to identify which components contribute most to classification.
    
    Args:
        model: The neural network model
        cache: ActivationCache with model activations
        class_idx: Class index to analyze (0=Neither, 1=Higgs, 2=W)
        top_k: Number of top contributors to return
        
    Returns:
        Dict of attribution scores by component
    """
    logits = cache["output"]
    batch_size, num_objects, num_classes = logits.shape
    
    attributions = {}
    
    # If no specific class is provided, use the predicted class
    if class_idx is None:
        class_idx = logits.argmax(dim=-1)
    elif isinstance(class_idx, int):
        class_idx = torch.full((batch_size, num_objects), class_idx, device=logits.device)
    
    # Compute attributions for each attention block
    for i in range(model.num_attention_blocks):
        # Get attention outputs
        attn_weights = cache[f"block_{i}_attention"]["attn_weights"]
        attn_output = cache[f"block_{i}_attention"]["OLD_attn_output"]
        
        # Get the classifier weights for the specified class
        classifier_weights = model.classifier[0].weight[class_idx]  # shape: [batch, obj, hidden_dim]
        
        # Calculate attribution per attention head
        num_heads = attn_weights.shape[1]
        head_attributions = []
        
        # Reshape attention output to separate heads
        # For standard nn.MultiheadAttention, we need to infer head dimension
        hidden_dim = attn_output.shape[-1]
        head_dim = hidden_dim // num_heads
        
        for h in range(num_heads):
            # Extract this head's contribution
            # This is approximate as we don't have direct access to per-head outputs
            head_slice = slice(h * head_dim, (h + 1) * head_dim)
            head_output = attn_output[..., head_slice]
            
            # Calculate attribution (dot product with classifier weights)
            head_attribution = torch.sum(head_output * classifier_weights[..., head_slice], dim=-1)
            head_attributions.append(head_attribution)
        
        attributions[f"block_{i}_attention"] = torch.stack(head_attributions, dim=0)
    
    # Aggregate attributions across batches for analysis
    aggregated = {}
    for key, attr in attributions.items():
        # Average across batch and objects
        aggregated[key] = attr.mean(dim=(1, 2)).cpu().numpy()
    
    # Find top contributors
    all_attrs = []
    for key, attrs in aggregated.items():
        for i, attr in enumerate(attrs):
            all_attrs.append((f"{key}_head_{i}", attr))
    
    # Sort by absolute attribution
    all_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "top_components": all_attrs[:top_k],
        "all_attributions": attributions,
        "aggregated": aggregated
    }


def direct_logit_attribution(model, cache, class_idx=None, top_k=5):
    """
    Perform direct logit attribution to identify which components contribute most to classification.
    
    Args:
        model: The neural network model
        cache: ActivationCache with model activations
        class_idx: Class index to analyze (0=Neither, 1=Higgs, 2=W)
        top_k: Number of top contributors to return
        
    Returns:
        Dict of attribution scores by component
    """
    logits = cache["output"]
    batch_size, num_objects, num_classes = logits.shape
    
    attributions = {}
    
    # If no specific class is provided, use the predicted class
    if class_idx is None:
        class_idx = logits.argmax(dim=-1)
    elif isinstance(class_idx, int):
        class_idx = torch.full((batch_size, num_objects), class_idx, device=logits.device)
    
    # Compute attributions for each attention block
    for i in range(model.num_attention_blocks):
        # Get attention outputs
        attn_weights = cache[f"block_{i}_attention"]["attn_weights_per_head"]
        attn_output = cache[f"block_{i}_attention"]["attn_output_per_head"]
        
        # Get the classifier weights for the specified class
        classifier_weights = model.classifier[0].weight[class_idx]  # shape: [batch, obj, hidden_dim]
        
        # Calculate attribution per attention head
        num_heads = attn_weights.shape[1]
        head_attributions = []
        
        # Reshape attention output to separate heads
        # For standard nn.MultiheadAttention, we need to infer head dimension
        hidden_dim = attn_output.shape[-1]
        
        for h in range(num_heads):
            # Extract this head's contribution
            # Calculate attribution (dot product with classifier weights)
            head_attribution = torch.sum(attn_output[:, h, :, :] * classifier_weights, dim=-1)
            head_attributions.append(head_attribution)
        
        attributions[f"block_{i}_attention"] = torch.stack(head_attributions, dim=0)
    
    # Aggregate attributions across batches for analysis
    aggregated = {}
    for key, attr in attributions.items():
        # Average across batch and objects
        aggregated[key] = attr.mean(dim=(1, 2)).cpu().numpy()
    
    # Find top contributors
    all_attrs = []
    for key, attrs in aggregated.items():
        for i, attr in enumerate(attrs):
            all_attrs.append((f"{key}_head_{i}", attr))
    
    # Sort by absolute attribution
    all_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "top_components": all_attrs[:top_k],
        "all_attributions": attributions,
        "aggregated": aggregated
    }



def analyze_attention_patterns(cache, block_idx=0, threshold=0.7):
    """
    Analyze attention patterns to find specialized attention heads.
    
    Args:
        cache: ActivationCache with model activations
        block_idx: Which attention block to analyze
        threshold: Correlation threshold for identifying specialized heads
        
    Returns:
        Dict of attention head patterns and their correlations with object types
    """
    attn_weights = cache[f"block_{block_idx}_attention"]["attn_weights"]
    
    # Get shape information
    batch_size, num_heads, num_queries, num_keys = attn_weights.shape
    
    # Average over batch dimension for analysis
    avg_attention = attn_weights.mean(dim=0).cpu().numpy()
    
    # Analyze each head's attention pattern
    head_patterns = {}
    for h in range(num_heads):
        head_pattern = avg_attention[h]
        head_patterns[f"head_{h}"] = head_pattern
    
    return {
        "attention_patterns": head_patterns,
        "raw_attention": attn_weights
    }

def plot_attention_heatmap(attention_patterns, head_idx, title=None):
    """Plot attention heatmap for a specific head."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_patterns[f"head_{head_idx}"], 
                cmap="viridis", 
                xticklabels=list(range(attention_patterns[f"head_{head_idx}"].shape[1])),
                yticklabels=list(range(attention_patterns[f"head_{head_idx}"].shape[0])))
    plt.xlabel("Key Position (attended to)")
    plt.ylabel("Query Position (attending)")
    plt.title(title or f"Attention Pattern for Head {head_idx}")
    plt.tight_layout()
    plt.show()

def analyze_object_type_attention(model, cache, object_types, padding_token, ret_dict=False, combine_elec_and_muon=False, exclude_self=False):
    """
    Analyze how different object types attend to each other.
    
    Args:
        model: The neural network model
        cache: ActivationCache with model activations
        object_types: Tensor of object types [batch, object]
        
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
        attn_weights = cache[f"block_{block_idx}_attention"]["attn_weights"]
        
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
                        not_diag_mask = (~(torch.eye(object_types.shape[-1]).to(bool))).unsqueeze(0) # [1 query key]
                    else:
                        not_diag_mask = torch.ones_like(key_mask) # [batch, 1, key]

                    # Apply masks and compute average attention
                    masked_attn = head_attn * query_mask * key_mask * not_diag_mask
                    
                    if 0: # Old method, used up to 2nd April 2025
                        # Normalize by the number of valid entries
                        valid_entries = (query_mask * key_mask).sum()
                        if valid_entries > 0:
                            avg_attn = masked_attn.sum() / valid_entries
                            if ret_dict:
                                type_attention[f"{query_type.item()}->{key_type.item()}"] = avg_attn.item()
                            else:
                                type_attention[query_type.item(), key_type.item()] = avg_attn.item()
                        else:
                            if ret_dict:
                                type_attention[f"{query_type.item()}->{key_type.item()}"] = 0
                            else:
                                type_attention[query_type.item(), key_type.item()] = 0
                    else: # New method, different way of averaging, let's see if it's better?
                        # Sum over the keys to get the total attention paid TO this type of object
                        # Then have to normalise by how many of these objects appear in TOTAL.
                        # This is because if the query object pays 100% attention to the key, the key dimension
                        # sum will equal 1 for each batch, query position, but there will be some multiple of these
                        # depending on how many of these query-type objects exist in total in our batch
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


def current_attn_detector(model, cache: ActivationCache) -> list[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    attn_heads = []
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[layer].self_attention.num_heads):
            attention_pattern = cache[f"block_{layer}_attention"]["attn_weights"][:,head]
            # take avg of diagonal elements
            score = (attention_pattern.diagonal(dim1=-2, dim2=-1) * (attention_pattern.diagonal(dim1=-2, dim2=-1)!=0)).sum() / ((attention_pattern.diagonal(dim1=-2, dim2=-1))!=0).sum()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def nanstd(input_tensor, dim=None, keepdim=False):
    mask = ~torch.isnan(input_tensor) # Mask NaN values
    input_tensor = torch.where(mask, input_tensor, torch.zeros_like(input_tensor)) # Replace NaNs with 0 for computation purposes
    count = mask.sum(dim=dim, keepdim=keepdim) # Compute the mean while ignoring NaNs
    mean = (input_tensor * mask).sum(dim=dim, keepdim=keepdim) / count
    variance = ((input_tensor - mean) ** 2 * mask).sum(dim=dim, keepdim=keepdim) / count # Compute variance while ignoring NaNs
    std = torch.sqrt(variance)    # Standard deviation is the square root of the variance
    return std

def likelyCandidate_attn_detector(model, cache: ActivationCache, candidate_direction:int, padding_token:int, object_types_to_include:Optional[list]=None) -> list[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be attending to 
    likely candidates *at this point in the model* for a particular target object (ie W/Higgs candidates)
    Reminder: candidate direction = 0 for not-inlcuded, 1 for included in SM higgs, 2 for included in W
    object_types_to_include is a list of object types included as queries; all objects (except padding) are 
        included in keys for the correlation calculation
    '''

    candidate_direction = model.classifier[0].weight[candidate_direction] # The vector in model space for the direction we want to maximise

    attn_heads = []
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[layer].self_attention.num_heads):
            attention_pattern = cache[f"block_{layer}_attention"]["attn_weights"][:,head] # Shape [batch object_query object_key]
            layer_inputs = cache[f"block_{layer}_attention"]['input'][0] # There are 3 entries to the inpusts list, but they are all the same
            layer_input_logits_in_candidate_direction = einops.einsum(layer_inputs, candidate_direction, 'batch object dmodel, dmodel-> batch object') # Shape [batch object]

            # Now actually calculate the correlation between attention pattern and pre-layer logits in candidate direction
            # We take the correlation of attn_pattern[batch, object_query, :] with layer_inputs_logits_in...[batch, :].
            # This results in a tensor of shape [batch, object_query]. We then take the mean of this, only over elements which pass the object_type mask (for now, this is just object_type!=0, but we may want to loop over object types later)
            a = attention_pattern
            if 0: # Test; make it either highly correlated or random to see what happens
                if 0: # Highly correlated
                    a = einops.repeat(layer_input_logits_in_candidate_direction, 'b o -> b k o', k=15)
                    a = a + torch.randn_like(a) * 0.2
                else: # Highly uncorrelated
                    a = einops.repeat(layer_input_logits_in_candidate_direction, 'b o -> b k o', k=15)
                    a = torch.randn_like(a)
            # Normalize `a` along the last dimension (object_k)
            if 1: # Have to ignore the padding
                a[cache['type_embedding']['input'][0]==padding_token] = torch.nan
                a_mean = a.nanmean(dim=-1, keepdim=True)
                a_std = nanstd(a, dim=-1, keepdim=True)
                # Send nan's back to 0
                a[torch.isnan(a)] = 0
            else:
                a_mean = a.mean(dim=-1, keepdim=True)
                a_std = a.std(dim=-1, keepdim=True)
            a_normalized = (a - a_mean) / (a_std + 1e-6)  # Add epsilon to avoid division by zero

            b = layer_input_logits_in_candidate_direction
            # Normalize `b` along the last dimension (object_k)
            if 1: # Have to ignore the padding
                b[cache['type_embedding']['input'][0]==padding_token] = torch.nan
                b_mean = b.nanmean(dim=-1, keepdim=True)
                b_std = nanstd(b, dim=-1, keepdim=True)
                # Send nan's back to 0
                b[torch.isnan(b)] = 0
            else:
                b_mean = b.mean(dim=-1, keepdim=True)
                b_std = b.std(dim=-1, keepdim=True)
            b_normalized = (b - b_mean) / (b_std + 1e-6)  # Add epsilon to avoid division by zero
            if 0: # Old, normalised just by the total possible number of key objects
                correlation = torch.einsum('boi,bi->bo', a_normalized, b_normalized) / (a_normalized.shape[1] - 1)
            else: # Better, correct (?), normalised by the actaul number of non-padding key objects
                correlation = torch.einsum('boi,bi->bo', a_normalized, b_normalized) / ((cache['type_embedding']['input'][0]!=padding_token).sum(dim=1, keepdim=True) - 1)
            
            if object_types_to_include is None:
                object_mask = (cache['type_embedding']['input'][0]!=padding_token) # Shape [batch object]
            else:
                object_mask = torch.isin(cache['type_embedding']['input'][0], torch.tensor(object_types_to_include)) # Shape [batch object]
            
            # Remove nans from this so that the sum doesn't get messed up
            correlation[~object_mask] = 0 # Could in theory just look for nans, or just look for object==padding_token, but do this for now
            score = (einops.einsum(correlation * object_mask, 'batch object ->') / einops.einsum(object_mask, 'batch object ->')) # Take mean of NON-padding objects

            print(f"{layer}.{head}: {score}")
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


def close_in_phi_detector(model, cache: ActivationCache, x, object_types, padding_token) -> list[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to 
    detectors for particles which are close in azimuthal angle
    '''
    attn_heads = []
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[layer].self_attention.num_heads):
            attention_pattern = cache[f"block_{layer}_attention"]["attn_weights"][:,head]
            # take avg of diagonal elements
            _, _, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
            wefwef
            
            deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object 1') - einops.rearrange(phi, 'batch object -> batch 1 object'))
            deltaPhisInRange = torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi
            absDeltaPhis = torch.abs(deltaPhisInRange)
            
            score = (attention_pattern.diagonal(dim1=-2, dim2=-1) * (attention_pattern.diagonal(dim1=-2, dim2=-1)!=0)).sum() / ((attention_pattern.diagonal(dim1=-2, dim2=-1))!=0).sum()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


def angular_separation_detector(model, cache: ActivationCache, x, object_types, padding_token) -> list[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to 
    detectors for particles which are close in azimuthal angle
    '''
    for _ in range(100):
        print("Maybe we don't want to check if deltaPhi is correlated to attention, but rather the response...")
    attn_heads = []
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[layer].self_attention.num_heads):
            attention_pattern = cache[f"block_{layer}_attention"]["attn_weights"][:,head]
            
            # Get the relevant parameters
            _, Eta, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
            deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object 1') - einops.rearrange(phi, 'batch object -> batch 1 object'))
            deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object 1') - einops.rearrange(Eta, 'batch object -> batch 1 object'))
            deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi)
            deltaEtas = deltaEtasOG.abs()

            # Now get the masks for the relevant objects
            object_counts = (object_types!=padding_token).sum(dim=-1).unsqueeze(-1)
            valid_object = ((object_types!=padding_token).unsqueeze(-1) & (object_types!=padding_token).unsqueeze(1))
            non_diagonal = (~(torch.eye(object_types.shape[1]).to(bool))).unsqueeze(0)
            consider_for_delta_phi = valid_object & non_diagonal
            # flat_consider_for_delta_phi = einops.rearrange(consider_for_delta_phi, 'batch query key -> batch (query key)')
            flat_consider_for_delta_phi = einops.rearrange(consider_for_delta_phi, 'batch query key -> (batch query key)')
            flat_attention = einops.rearrange(attention_pattern,'batch query key -> (batch query key)')
            flat_delta_phi = einops.rearrange(deltaPhis,'batch query key -> (batch query key)')
            flat_delta_eta = einops.rearrange(deltaEtas,'batch query key -> (batch query key)')

            
            # Now extract just a few samples of this (masked) dataset for training the symbolic regression and put them into numpy arrays
            X = np.concat([flat_delta_phi[flat_consider_for_delta_phi].cpu().numpy().reshape(-1,1),flat_delta_eta[flat_consider_for_delta_phi].cpu().numpy().reshape(-1,1)], axis=1)
            y = flat_attention[flat_consider_for_delta_phi].cpu().numpy()
            MAX_ntrain=1000
            X_test = X[MAX_ntrain:2*MAX_ntrain]
            y_test = y[MAX_ntrain:2*MAX_ntrain]
            X = X[:MAX_ntrain]
            y = y[:MAX_ntrain]

            # Now try and predict with SR
            est_pysr = PySRRegressor(
                population_size=500,
                niterations=20,
                binary_operators=["+", "*", "^"],
                unary_operators=[],
                constraints={'^': (-1, 1)},
                maxsize=20,
                parsimony=0.01,
                # procs=0.9,
                ncyclesperiteration=500,
                # verbosity=1,
                random_state=0,
                deterministic=True,
                procs=0,
                # optimize_probability=0.7,
                model_selection='accuracy',
                verbosity=0,
            )
            est_pysr.fit(X, y)
            
            # Print the best found expression
            print(est_pysr.sympy())
            # Make predictions
            y_pred = est_pysr.predict(X)
            # Calculate R-squared score
            r_squared = est_pysr.score(X, y)
            r_squared_test = est_pysr.score(X_test, y_test)
            print(f"R-squared score: {r_squared} (Test: {r_squared_test})")

            if r_squared > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads





def angular_separation_detector_split_by_type(model, 
                                              cache: ActivationCache, 
                                              x, 
                                              object_types, 
                                              padding_token,
                                              layers=None,
                                              heads=None,
                                              query_types=None,
                                              key_types=None,
                                              ) -> list[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to 
    detectors for particles which are close in azimuthal angle
    '''
    attn_heads = []
    scores = {}
    # for layer in range(len(model.attention_blocks)):
    if layers is None:
        layers = range(len(model.attention_blocks))
    if heads is None:
        heads = range(model.attention_blocks[0].self_attention.num_heads)
    for layer in layers:
        scores[layer] = {}
        for head in heads:
            scores[layer][head] = {}
            attention_pattern = cache[f"block_{layer}_attention"]["attn_weights"][:,head]
            
            # Get the relevant parameters
            _, Eta, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
            deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object 1') - einops.rearrange(phi, 'batch object -> batch 1 object'))
            deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object 1') - einops.rearrange(Eta, 'batch object -> batch 1 object'))
            deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi)
            deltaEtas = deltaEtasOG.abs()

            
            # Now get the masks for the relevant objects
            if query_types is None:
                unique_types = torch.unique(object_types)
                query_types = unique_types
            if key_types is None:
                unique_types = torch.unique(object_types)
                key_types = unique_types
            for query_type in query_types:
                scores[layer][head][query_type] = {}
                for key_type in key_types:
                    print(f"Layer/head/query/key: {layer}/{head}/{query_type}/{key_type}")
                    scores[layer][head][query_type][key_type] = {}
                    # Skip padding tokens
                    if query_type == (padding_token) or key_type == (padding_token):
                        continue
                        
                    # Create masks for the query and key object types
                    query_mask = einops.repeat((object_types == query_type), 'batch query -> batch query num_objs', num_objs=object_types.shape[-1])  # [batch, query, 1]
                    key_mask =  einops.repeat((object_types == key_type), 'batch key -> batch num_objs key', num_objs=object_types.shape[-1])  # [batch, 1, key]
                    
                    # Apply masks and compute average attention
                    object_counts = (object_types!=padding_token).sum(dim=-1).unsqueeze(-1)
                    valid_object = ((object_types!=padding_token).unsqueeze(-1) & (object_types!=padding_token).unsqueeze(1))
                    non_diagonal = (~(torch.eye(object_types.shape[1]).to(bool))).unsqueeze(0)
                    if 1: # Because I think here we do want to check (eg. small-R looking for other close small-R) but NOT looking at self
                        consider_for_delta_phi = valid_object & non_diagonal & query_mask & key_mask
                    else:
                        consider_for_delta_phi = valid_object & query_mask & key_mask
                    # flat_consider_for_delta_phi = einops.rearrange(consider_for_delta_phi, 'batch query key -> batch (query key)')
                    flat_consider_for_delta_phi = einops.rearrange(consider_for_delta_phi, 'batch query key -> (batch query key)')
                    flat_attention = einops.rearrange(attention_pattern,'batch query key -> (batch query key)')
                    flat_delta_phi = einops.rearrange(deltaPhis,'batch query key -> (batch query key)')
                    flat_delta_eta = einops.rearrange(deltaEtas,'batch query key -> (batch query key)')

                    
                    # Now extract just a few samples of this (masked) dataset for training the symbolic regression and put them into numpy arrays
                    X = np.concat([flat_delta_phi[flat_consider_for_delta_phi].cpu().numpy().reshape(-1,1),flat_delta_eta[flat_consider_for_delta_phi].cpu().numpy().reshape(-1,1)], axis=1)
                    y = flat_attention[flat_consider_for_delta_phi].cpu().numpy()
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    MAX_ntrain_orig=1000
                    MAX_ntrain=min(MAX_ntrain_orig, len(X)//2)
                    if MAX_ntrain<MAX_ntrain_orig:
                        print(f"WARNING: MAX_ntrain = {MAX_ntrain}<{MAX_ntrain_orig}")
                        if MAX_ntrain<100:
                            print(f"WARNING REAL WARNING: MAX_ntrain = {MAX_ntrain}<10 so I'm skipping")
                            continue

                        # assert(False)
                    X_test = X[MAX_ntrain:2*MAX_ntrain]
                    y_test = y[MAX_ntrain:2*MAX_ntrain]
                    X = X[:MAX_ntrain]
                    y = y[:MAX_ntrain]


                    if 0:
                        scaler = StandardScaler()
                        X = scaler.fit_transform(X)
                        scaler = StandardScaler()
                        y = scaler.fit_transform(y.reshape(-1,1)).flatten()

                    # Now try and predict with SR
                    est_pysr = PySRRegressor(
                        population_size=2000,
                        niterations=10,
                        binary_operators=["+", "*", "^", "-"],
                        unary_operators=[],
                        constraints={'^': (-1, 1)},
                        complexity_of_variables=2,
                        maxsize=20,
                        parsimony=0.01,
                        # procs=0.9,
                        ncyclesperiteration=500,
                        # verbosity=1,
                        verbosity=0,
                        random_state=0,
                        deterministic=False,
                        procs=4,
                        # optimize_probability=0.7,
                        model_selection='accuracy',
                    )
                    est_pysr.fit(X, y)
                    
                    # Print the best found expression
                    print(est_pysr.sympy())
                    # Make predictions
                    y_pred = est_pysr.predict(X)
                    # Calculate R-squared score
                    r_squared = est_pysr.score(X, y)
                    r_squared_test = est_pysr.score(X_test, y_test)
                    print(f"R-squared score: {r_squared} (Test: {r_squared_test})")

                    if r_squared > 0.5:
                        attn_heads.append(f"{layer}.{head} ({query_type}->{key_type})")
                    scores[layer][head][query_type][key_type]['score_train'] = r_squared
                    scores[layer][head][query_type][key_type]['score_test'] = r_squared_test
                    scores[layer][head][query_type][key_type]['eq'] = est_pysr.sympy()
    return attn_heads, scores






# %%
def get_direct_logit_attribution(model, # The model to test
                                 cache,
                                 truth_inclusion,
                                #  pred_inclusion, # Can get this from logits
                                 object_types, 
                                 true_incl:list,
                                 type_incl:list,
                                 class_idx,
                                 include_mlp=True,
                                 include_direct_from_embedding=True,
                                ):
    """
    Perform direct logit attribution to identify which components contribute most to classification.
    
    Args:
        model: The neural network model
        cache: ActivationCache with model activations
        class_idx: Class index to analyze (0=Neither, 1=Higgs, 2=W)
        top_k: Number of top contributors to return
        
    Returns:
        Dict of attribution scores by component
    """
    logits = cache["output"]
    batch_size, num_objects, num_classes = logits.shape
    
    attributions = {}

    mask = ((torch.isin(truth_inclusion, torch.Tensor(true_incl))) & 
            (torch.isin(object_types, torch.Tensor(type_incl)))
            ).to(logits.device)

    class_idx = torch.full((batch_size, num_objects), class_idx, device=logits.device)
    # Get the classifier weights for the specified class
    classifier_weights = model.classifier[0].weight[class_idx]  # shape: [batch, object, hidden_dim]

    if include_direct_from_embedding:
        # Compute attention for directly encoded/decoded
        attributions["object_net"] = einops.einsum(cache['object_net']['output'], classifier_weights, 'batch object dmod, batch object dmod -> batch object')

    # Compute attributions for each attention block
    for i in range(model.num_attention_blocks):
        # Get attention outputs
        attn_weights = cache[f"block_{i}_attention"]["attn_weights"]
        if 0: # Use old method of separating out attention output; doesn't work since the heads are averaged during projection
            for _ in range(10):
                print("WARNING: using old method which won't attribute to the heads correctly")
            attn_output = cache[f"block_{i}_attention"]["OLD_attn_output"]
            
            
            # Calculate attribution per attention head
            num_heads = attn_weights.shape[1]
            head_attributions = []
            
            # Reshape attention output to separate heads
            # For standard nn.MultiheadAttention, we need to infer head dimension
            hidden_dim = attn_output.shape[-1]
            head_dim = hidden_dim // num_heads
            
            for h in range(num_heads):
                # Extract this head's contribution
                # This is approximate as we don't have direct access to per-head outputs
                head_slice = slice(h * head_dim, (h + 1) * head_dim)
                head_output = attn_output[..., head_slice]
                
                # Calculate attribution (dot product with classifier weights)
                head_attribution = torch.sum(head_output * classifier_weights[..., head_slice], dim=-1)
                head_attributions.append(head_attribution)
        else:
            attn_output = cache[f"block_{i}_attention"]["attn_output_per_head"]
            num_heads = attn_weights.shape[1]
            head_attributions = []
            for h in range(num_heads): # Calculate attribution per attention head
                # Extract this head's contribution & calculate attribution (dot product with classifier weights)
                head_attribution = torch.sum(attn_output[:, h, :, :] * classifier_weights, dim=-1)
                head_attributions.append(head_attribution)
        
        if include_mlp:
            mlp_output = cache[f"block_{i}_post_attention"]['output']
        
        attributions[f"block_{i}_attention"] = torch.stack(head_attributions, dim=0)

        if include_mlp:
            # Get MLP contributions
            attributions[f"block_{i}_mlp"] = einops.einsum(mlp_output, classifier_weights, 'batch object dmod, batch object dmod -> batch object')
    
    
    # Aggregate attributions across batches for analysis
    aggregated = {}
    for key, attr in attributions.items():
        # Average across batch and objects
        aggregated[key] = einops.einsum(attr* mask, '... batch object -> ...') / einops.einsum(mask, 'batch object ->')
        # aggregated[key] = attr.mean(dim=(1, 2)).cpu().numpy()
    
    
    conts = np.zeros((model.num_attention_blocks+include_direct_from_embedding, num_heads + include_mlp))*np.nan
    if include_direct_from_embedding:
        conts[0,0] = aggregated["object_net"].detach().cpu()
    for i in range(model.num_attention_blocks):
        conts[i+include_direct_from_embedding,:num_heads] = aggregated[f"block_{i}_attention"].detach().cpu().numpy()
        if include_mlp:
            conts[i+include_direct_from_embedding,-1] = aggregated[f"block_{i}_mlp"].detach().cpu()

    return conts

def plot_logit_attributions(model, 
                            cache,
                            truth_inclusion,
                            object_types, 
                            true_incl:list,
                            type_incl:list,
                            include_mlp=True,
                            TRANSPOSE=False,
                            title=None,
                            include_direct_from_embedding=True,
):
    assert(isinstance(true_incl, list)) # Just because this caused problems one time with empty plots but no fail
    assert(isinstance(type_incl, list)) # Similar check
    if TRANSPOSE:
        fig = plt.figure(figsize=(9,3))
    else:
        fig = plt.figure(figsize=(8,4))
    vmax=0
    r={}
    for class_idx in range(3):
        r[class_idx] = get_direct_logit_attribution(model, 
                                    cache, 
                                    truth_inclusion, 
                                    object_types,
                                    true_incl,
                                    type_incl,
                                    class_idx,
                                    include_mlp=include_mlp,
                                    include_direct_from_embedding=include_direct_from_embedding,
        )
        plt.subplot(1,3,class_idx+1)
        vmax=max(vmax, np.nanmax(abs(r[class_idx])))
    for class_idx in range(3):
        plt.subplot(1,3,class_idx+1)
        cmap = mpl.colormaps.get_cmap('PiYG')
        cmap.set_bad(color='white')
        if TRANSPOSE:
            plt.imshow(r[class_idx].transpose(), cmap=cmap, vmin=-vmax, vmax=vmax)
            # hatch = plt.contourf(np.isnan(r[class_idx].transpose()), 1, hatches=['\\', '//'], alpha=0, origin="lower", extent=(-0.5, r[class_idx].shape[0]-0.5, r[class_idx].shape[1]-0.5, -0.5))
        else:
            plt.imshow(r[class_idx], cmap=cmap, vmin=-vmax, vmax=vmax)
            # plt.imshow(r[class_idx]+vmax, cmap=cmap, norm=mpl.colors.LogNorm(vmin=2, vmax=vmax))
            # hatch = plt.pcolor(np.isnan(r[class_idx]), hatch='///', alpha=0)
            # hatch = plt.contourf(np.isnan(r[class_idx]), 1, hatches=['\\', '//'], alpha=0, origin="lower", extent=(-0.5, r[class_idx].shape[1]-0.5, r[class_idx].shape[0]-0.5, -0.5))
            for i in range(r[class_idx].shape[0]):
                for j in range(r[class_idx].shape[1]):
                    if np.isnan(r[class_idx])[i, j]:
                        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch='///', alpha=1, linewidth=0)
                        plt.gca().add_patch(rect)
        plt.title(f"Class {class_idx} DLA")

        if TRANSPOSE and (class_idx==2):
            plt.colorbar(fraction=0.046, pad=0.04)
        if (not TRANSPOSE) and (class_idx==2):
            plt.colorbar(fraction=0.066, pad=0.04)
        # TODO  Could tidy this up...
        if (not TRANSPOSE) and include_mlp:
            plt.xticks(range(r[class_idx].shape[0]+1), [f"Head {i}" for i in range(r[class_idx].shape[0])] + ['MLP'], rotation=85)
        if (not TRANSPOSE) and (not include_mlp):
            plt.xticks(range(r[class_idx].shape[1]), [f"Head {i}" for i in range(r[class_idx].shape[1])], rotation=85)
            if include_direct_from_embedding:
                plt.yticks(range(r[class_idx].shape[0]), ["Embedding"] + [f"Layer {i}" for i in range(r[class_idx].shape[0]-1)], rotation=5)
            else:
                print(f"{r[class_idx].shape=}")
                print(f"{r[class_idx]=}")
                plt.yticks(range(r[class_idx].shape[0]), [f"Layer {i}" for i in range(r[class_idx].shape[0])], rotation=5)
    if title is None:
        plt.suptitle(f"Logit attribution for true incl. {true_incl}, type {type_incl}")
    else:
        plt.suptitle(title)
    if not TRANSPOSE:
        fig.supxlabel(f"Posn in layer")
        fig.supylabel(f"Layer #")
    plt.tight_layout()
    return fig





# %%
def get_direct_logit_attributionMultiBatch(model, # The model to test
                                 dataloader,
                                 num_inputs,
                                 true_incl:list,
                                 type_incl:list,
                                 class_idx,
                                 include_mlp=True,
                                 include_direct_from_embedding=True,
                                 min_samples=1000,
                                 dsid_incl=None,
                                ):
    """
    Perform direct logit attribution to identify which components contribute most to classification.
    This version takes in the dataloader and runs until a set number of samples is reached.

    
    Args:
        model: The neural network model
        cache: ActivationCache with model activations
        class_idx: Class index to analyze (0=Neither, 1=Higgs, 2=W)
        top_k: Number of top contributors to return
        
    Returns:
        Dict of attribution scores by component
    """

    # Initialise lists of attributions for each component
    attributions = {}
    masks = []
    if include_direct_from_embedding:
        attributions.update({'object_net':[]})
    attributions.update({f"block_{i}_attention":[] for i in range(model.num_attention_blocks)})
    if include_mlp:
        attributions.update({f"block_{i}_mlp":[] for i in range(model.num_attention_blocks)})

    num_samples_processed = 0
    num_batches_processed = 0
    dataloader._reset_indices()
    num_batches_total = len(dataloader)
    # Loop over the dataloader to get the info we need
    while ((num_samples_processed<=min_samples) and (num_batches_processed < num_batches_total)):
        # Get the samples
        batch = next(dataloader)
        num_batches_processed += 1
        x, y, _, types, dsids, _, _, _, _ = batch.values()
        cache = extract_all_activations(model, x[...,:num_inputs], types)
        # Run the model to get the outputs/cache
        logits = cache["output"]
        batch_size, num_objects, num_classes = logits.shape
    
        if dsid_incl is None:
            dsid_sel = einops.repeat(torch.ones_like(dsids==0), 'batch -> batch object', object=x.shape[1])
        else:
            dsid_sel = einops.repeat(torch.isin(dsids, torch.Tensor(dsid_incl)), 'batch -> batch object', object=x.shape[1])
        masks.append(
            (
                (torch.isin(x[...,-1], torch.Tensor(true_incl))) & 
                (torch.isin(types, torch.Tensor(type_incl))) & 
                dsid_sel
            ).to(logits.device)
            )
        num_samples_processed += masks[-1].any(dim=-1).sum().item()

        class_idx_t = torch.full((batch_size, num_objects), class_idx, device=logits.device)
        # Get the classifier weights for the specified class
        classifier_weights = model.classifier[0].weight[class_idx_t]  # shape: [batch, object, hidden_dim]

        if include_direct_from_embedding:
            # Compute attention for directly encoded/decoded
            attributions["object_net"].append(einops.einsum(cache['object_net']['output'], classifier_weights, 'batch object dmod, batch object dmod -> batch object'))

        # Compute attributions for each attention block
        for i in range(model.num_attention_blocks):
            # Get attention outputs
            attn_weights = cache[f"block_{i}_attention"]["attn_weights"]
            if 0: # Use old method of separating out attention output; doesn't work since the heads are averaged during projection
                for _ in range(10):
                    print("WARNING: using old method which won't attribute to the heads correctly")
                attn_output = cache[f"block_{i}_attention"]["OLD_attn_output"]
                
                
                # Calculate attribution per attention head
                num_heads = attn_weights.shape[1]
                head_attributions = []
                
                # Reshape attention output to separate heads
                # For standard nn.MultiheadAttention, we need to infer head dimension
                hidden_dim = attn_output.shape[-1]
                head_dim = hidden_dim // num_heads
                
                for h in range(num_heads):
                    # Extract this head's contribution
                    # This is approximate as we don't have direct access to per-head outputs
                    head_slice = slice(h * head_dim, (h + 1) * head_dim)
                    head_output = attn_output[..., head_slice]
                    
                    # Calculate attribution (dot product with classifier weights)
                    head_attribution = torch.sum(head_output * classifier_weights[..., head_slice], dim=-1)
                    head_attributions.append(head_attribution)
            else:
                attn_output = cache[f"block_{i}_attention"]["attn_output_per_head"]
                num_heads = attn_weights.shape[1]
                head_attributions = []
                for h in range(num_heads): # Calculate attribution per attention head
                    # Extract this head's contribution & calculate attribution (dot product with classifier weights)
                    head_attribution = torch.sum(attn_output[:, h, :, :] * classifier_weights, dim=-1)
                    head_attributions.append(head_attribution)
        
            if include_mlp:
                mlp_output = cache[f"block_{i}_post_attention"]['output']
            
            attributions[f"block_{i}_attention"].append(torch.stack(head_attributions, dim=0))

            if include_mlp:
                # Get MLP contributions
                attributions[f"block_{i}_mlp"].append(einops.einsum(mlp_output, classifier_weights, 'batch object dmod, batch object dmod -> batch object'))    
    
    print(f"Calculating with {num_samples_processed} samples")
    # Cat the attirbutions across all batches
    for k in attributions.keys():
        attributions[k] = torch.concatenate(attributions[k], dim=-2)
    mask = torch.concatenate(masks, dim=0)
    
    
    # Aggregate attributions across batches for analysis
    aggregated = {}
    for key, attr in attributions.items():
        # Average across batch and objects
        aggregated[key] = einops.einsum(attr* mask, '... batch object -> ...') / einops.einsum(mask, 'batch object ->')
        # aggregated[key] = attr.mean(dim=(1, 2)).cpu().numpy()
    
    
    conts = np.zeros((model.num_attention_blocks+include_direct_from_embedding, num_heads + include_mlp))*np.nan
    if include_direct_from_embedding:
        conts[0,0] = aggregated["object_net"].detach().cpu()
    for i in range(model.num_attention_blocks):
        conts[i+include_direct_from_embedding,:num_heads] = aggregated[f"block_{i}_attention"].detach().cpu().numpy()
        if include_mlp:
            conts[i+include_direct_from_embedding,-1] = aggregated[f"block_{i}_mlp"].detach().cpu()

    return conts

def plot_logit_attributionsMultiBatch(model, 
                            dataloader,
                            num_inputs:int,
                            true_incl:list,
                            type_incl:list,
                            include_mlp=True,
                            TRANSPOSE=False,
                            title=None,
                            include_direct_from_embedding=True,
                            min_samples=1000,
                            dsid_incl=None,
):
    assert(isinstance(true_incl, list)) # Just because this caused problems one time with empty plots but no fail
    assert(isinstance(type_incl, list)) # Similar check
    if TRANSPOSE:
        fig = plt.figure(figsize=(9,3))
    else:
        fig = plt.figure(figsize=(8,4))
    vmax=0
    r={}
    for class_idx in range(3):
        r[class_idx] = get_direct_logit_attributionMultiBatch(model, 
                                    dataloader, 
                                    num_inputs,
                                    true_incl,
                                    type_incl,
                                    class_idx,
                                    include_mlp=include_mlp,
                                    include_direct_from_embedding=include_direct_from_embedding,
                                    min_samples=min_samples,
                                    dsid_incl=dsid_incl,
        )
        plt.subplot(1,3,class_idx+1)
        vmax=max(vmax, np.nanmax(abs(r[class_idx])))
    for class_idx in range(3):
        plt.subplot(1,3,class_idx+1)
        cmap = mpl.colormaps.get_cmap('PiYG')
        cmap.set_bad(color='white')
        if TRANSPOSE:
            plt.imshow(r[class_idx].transpose(), cmap=cmap, vmin=-vmax, vmax=vmax)
            # hatch = plt.contourf(np.isnan(r[class_idx].transpose()), 1, hatches=['\\', '//'], alpha=0, origin="lower", extent=(-0.5, r[class_idx].shape[0]-0.5, r[class_idx].shape[1]-0.5, -0.5))
        else:
            plt.imshow(r[class_idx], cmap=cmap, vmin=-vmax, vmax=vmax)
            # plt.imshow(r[class_idx]+vmax, cmap=cmap, norm=mpl.colors.LogNorm(vmin=2, vmax=vmax))
            # hatch = plt.pcolor(np.isnan(r[class_idx]), hatch='///', alpha=0)
            # hatch = plt.contourf(np.isnan(r[class_idx]), 1, hatches=['\\', '//'], alpha=0, origin="lower", extent=(-0.5, r[class_idx].shape[1]-0.5, r[class_idx].shape[0]-0.5, -0.5))
            for i in range(r[class_idx].shape[0]):
                for j in range(r[class_idx].shape[1]):
                    if np.isnan(r[class_idx])[i, j]:
                        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch='///', alpha=1, linewidth=0)
                        plt.gca().add_patch(rect)
        plt.title(f"Class {class_idx} DLA")

        if TRANSPOSE and (class_idx==2):
            plt.colorbar(fraction=0.046, pad=0.04)
        if (not TRANSPOSE) and (class_idx==2):
            plt.colorbar(fraction=0.066, pad=0.04)
        # TODO  Could tidy this up...
        if (not TRANSPOSE) and include_mlp:
            plt.xticks(range(r[class_idx].shape[0]+1), [f"Head {i}" for i in range(r[class_idx].shape[0])] + ['MLP'], rotation=85)
        if (not TRANSPOSE) and (not include_mlp):
            plt.xticks(range(r[class_idx].shape[1]), [f"Head {i}" for i in range(r[class_idx].shape[1])], rotation=85)
            if include_direct_from_embedding:
                plt.yticks(range(r[class_idx].shape[0]), ["Embedding"] + [f"Layer {i}" for i in range(r[class_idx].shape[0]-1)], rotation=5)
            else:
                print(f"{r[class_idx].shape=}")
                print(f"{r[class_idx]=}")
                plt.yticks(range(r[class_idx].shape[0]), [f"Layer {i}" for i in range(r[class_idx].shape[0])], rotation=5)
    if title is None:
        plt.suptitle(f"Logit attribution for true incl. {true_incl}, type {type_incl}")
    else:
        plt.suptitle(title)
    if not TRANSPOSE:
        fig.supxlabel(f"Posn in layer")
        fig.supylabel(f"Layer #")
    plt.tight_layout()
    return fig






















import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_param=0.05, beta=3.0):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self.sparsity_param = sparsity_param
        self.beta = beta
        
    def forward(self, x):
        encoded = self.activation(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def kl_divergence(self, rho_hat):
        """Calculate KL divergence to enforce sparsity"""
        rho = torch.ones_like(rho_hat) * self.sparsity_param
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return kl.sum()

def train_sparse_autoencoder(activations, hidden_dim=50, epochs=100, lr=1e-3, sparsity_param=0.05, beta=3.0):
    """
    Train a sparse autoencoder on model activations
    
    Args:
        activations: Tensor of shape [n_samples, feature_dim]
        hidden_dim: Number of hidden units (features to extract)
        epochs: Number of training epochs
        lr: Learning rate
        sparsity_param: Target activation frequency (lower = sparser)
        beta: Weight of sparsity penalty
        
    Returns:
        Trained autoencoder model
    """
    input_dim = activations.shape[1]
    model = SparseAutoencoder(input_dim, hidden_dim, sparsity_param, beta)
    
    # Normalize data
    mean = activations.mean(dim=0, keepdim=True)
    std = activations.std(dim=0, keepdim=True) + 1e-6
    normalized_activations = (activations - mean) / std
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        decoded, encoded = model(normalized_activations)
        
        # Calculate reconstruction loss
        loss_recon = mse_loss(decoded, normalized_activations)
        
        # Calculate sparsity loss
        rho_hat = encoded.mean(dim=0)  # Average activation of hidden units
        loss_sparse = model.kl_divergence(rho_hat)
        
        # Total loss
        if beta != 0:
            loss = loss_recon + beta * loss_sparse
        else: # I think this doesn't matter but just in case it matters
            loss = loss_recon
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.10f}, Recon: {loss_recon.item():.10f}, Sparse: {loss_sparse.item():.10f}")
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.10e}, Recon: {loss_recon.item():.10e}, Sparse: {loss_sparse.item():.10e}")
    
    return model, mean, std

def extract_features(autoencoder, activations, mean, std):
    """Extract features using a trained autoencoder"""
    normalized_activations = (activations - mean) / std
    with torch.no_grad():
        _, encoded = autoencoder(normalized_activations)
    return encoded

def analyze_features(encoded_features, top_k=10):
    """Find top activating examples for each feature"""
    feature_activations = {}
    for feature_idx in range(encoded_features.shape[1]):
        # Get indices of examples that most activate this feature
        activations = encoded_features[:, feature_idx].numpy()
        top_examples = np.argsort(activations)[-top_k:][::-1]
        feature_activations[feature_idx] = {
            'indices': top_examples,
            'values': activations[top_examples]
        }
    return feature_activations

def visualize_latent_space(encoded_features, labels=None, method='tsne'):
    """Visualize latent space using PCA or t-SNE"""
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
    
    reduced = reducer.fit_transform(encoded_features.detach().numpy())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels if labels is not None else 'blue', 
                         alpha=0.5, cmap='viridis',s=1)
    
    if labels is not None:
        plt.colorbar(scatter, label='Class')
    
    plt.title(f'Latent Space Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    return plt

def regress_results(y, X, n_train=2000, labels=None):
    if y.ndim==1:
        y = y.unsqueeze(-1)
    if X.shape[-1] == 10: # Assume
        variable_names = ['pT_0', 'Eta_0', 'phi_0', 'M_0', 'tag_0', 'pT_1', 'Eta_1', 'phi_1', 'M_1', 'tag_1']
    if X.shape[-1] == 8: # Assume
        variable_names = ['pT_0', 'Eta_0', 'phi_0', 'M_0', 'pT_1', 'Eta_1', 'phi_1', 'M_1']
    if X.shape[-1] == 5: # Assume
        variable_names = ['pT_0', 'Eta_0', 'phi_0', 'M_0', 'tag_0']
    if X.shape[-1] == 4: # Assume
        variable_names = ['pT_0', 'Eta_0', 'phi_0', 'M_0']
    else:
        variable_names = [f'X{i}' for i in range(X.shape[-1])]
    results = {}
    for feature_n in range(y.shape[1]):
        X_test = X[n_train:2*n_train]
        y_test = y[n_train:2*n_train, feature_n].flatten()
        X_train = X[:n_train]
        y_train = y[:n_train, feature_n].flatten()

        if 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            scaler = StandardScaler()
            y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()

        # Now try and predict with SR
        est_pysr = PySRRegressor(
            population_size=1000,
            niterations=50,
            binary_operators=["+", "*", "^", "-"],
            unary_operators=[],
            constraints={'^': (-1, 1)},
            complexity_of_variables=2,
            maxsize=40,
            parsimony=0.01,
            # procs=0.9,
            ncyclesperiteration=500,
            verbosity=1,
            # verbosity=0,
            random_state=0,
            deterministic=False,
            procs=4,
            # optimize_probability=0.7,
            model_selection='accuracy',
            variable_names=variable_names,
        )
        est_pysr.fit(X_train, y_train)
        
        # Print the best found expression
        print(est_pysr.sympy())
        # Make predictions
        y_pred = est_pysr.predict(X_train)
        # Calculate R-squared score
        r_squared = est_pysr.score(X_train, y_train)
        try:
            r_squared_test = est_pysr.score(X_test, y_test)
        except:
            r_squared_test = None
        print(f"R-squared score: {r_squared} (Test: {r_squared_test})")
        results[feature_n] = {}
        results[feature_n]['score_train'] = r_squared
        results[feature_n]['score_test'] = r_squared_test
        results[feature_n]['eq'] = est_pysr.sympy()
    return results

def run_sae_analysis(layer_activations, layer_key, batch_data, true_labels=None, hidden_dim=50):
    """Run full SAE analysis for a specific layer's activations"""
    print(f"\nAnalyzing layer: {layer_key}")
    assert (len(layer_activations.shape)==2)
    if true_labels is not None:
        assert (len(true_labels)==len(layer_activations))
    
    
    # Reshape activations if needed - assuming [batch, objects, features]
    batch_size, n_features = layer_activations.shape
    flattened_activations = layer_activations
    
    # Train the autoencoder
    print("Training sparse autoencoder...")
    autoencoder, mean, std = train_sparse_autoencoder(
        flattened_activations, 
        hidden_dim=hidden_dim, 
        epochs=10000,
        lr=2e-3,
        sparsity_param=0.05,
        # beta=1e-3
        beta=0
    )
    
    # Extract features
    encoded_features = extract_features(autoencoder, flattened_activations, mean, std)
    
    # Analyze features
    feature_info = analyze_features(encoded_features)
    
    # Visualize if labels provided
    if (true_labels is not None) and (encoded_features.shape[-1]>1):
        # Repeat labels for each object
        # repeated_labels = true_labels.unsqueeze(1).expand(-1, n_objects).reshape(-1)
        repeated_labels = true_labels
        plt = visualize_latent_space(encoded_features, repeated_labels)
        plt.savefig(f"latent_space_{layer_key}_autoencoderDim{encoded_features.shape[-1]}.png")
        plt.close()
    
    pysr_results = regress_results(encoded_features.detach().cpu().numpy(), batch_data.detach().cpu().numpy(), n_train=500)
    
    # Return the trained model and extracted features
    return {
        'autoencoder': autoencoder,
        'features': encoded_features,
        'feature_info': feature_info,
        'normalizers': (mean, std),
        'symbolic_regression': pysr_results,
    }

def interpret_features(autoencoder, feature_idx, input_dim):
    """Interpret what each feature is detecting based on decoder weights"""
    # Get the decoder weights for this feature
    with torch.no_grad():
        decoder_weights = autoencoder.decoder.weight[:, feature_idx].numpy()
    
    # Get indices of the highest magnitude weights (positive and negative)
    top_positive = np.argsort(decoder_weights)[-10:][::-1]
    top_negative = np.argsort(decoder_weights)[:10]
    
    print(f"Feature {feature_idx} interpretation:")
    print("Top positive connections:")
    for i, idx in enumerate(top_positive):
        print(f"  Input {idx}: {decoder_weights[idx]:.4f}")
    
    print("Top negative connections:")
    for i, idx in enumerate(top_negative):
        print(f"  Input {idx}: {decoder_weights[idx]:.4f}")
        
    return {
        'top_positive': top_positive,
        'top_negative': top_negative,
        'weights': decoder_weights
    }

# Example usage
def main_sae_analysis(model, data_loader, n_inputs, device='cuda', hidden_dim_sae=50):
    """Main function to run SAE analysis on selected layers"""
    results = {}
    
    # Layers of interest (based on your DLA findings)
    target_layers = [
        # 'object_net', 
        ('block_0_attention', 'attn_output_per_head', 1),  # Example - replace with your layers of interest
        ('block_0_attention', 'attn_unprojected_output_per_head', 1),  # Example - replace with your layers of interest
        # 'block_1_post_attention',  # Example - replace with your layers of interest
        # 'block_3_post_attention'
    ]
    print("WARNING: Hardcoded a bunch of stuff here")
    target_query_objects = [0] # Objects which we're looking at the activations of
    target_key_objects = [2] # Objects which we think might be interesting in interpreting the activations. We'll take the pt, eta, phi, m, tagInfo of these variables and try to use symbolic regression to get a decent closed for expression for what the extracted feature is calculating
    true_incl = [0,3] # Look at activations which are, in truth labelling, included in this particle. 3 is Wleptonic from H+, 2 is Whadronic from H+, 1 is SM Higgs from H+, 0 is none
    

    # Run analysis for each target layer
    for layer in target_layers:
        N_TRAIN = 10000 # min training samples for the SAE
        hidden_dim = model.object_net[0].out_features
        layer_activations = None
        layer_truths = None
        layer_potential_useful = None
        num_activs = 0
        data_loader._reset_indices()
        total_batches = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            print(f"Batch {batch_idx}/{total_batches}")
            x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
            x, types = x.to(device), types.to(device)
            # Extract activations
            cache = extract_all_activations(model, x[...,:n_inputs], types)
            if isinstance(layer, tuple):
                if layer[1] == 'attn_output_per_head':
                    is_attention=True
                    layer_num = int(layer[0].split('_')[1][0])
                    head_num = layer[2]
                    activations = cache[layer[0]][layer[1]] # Shape [batch head object dmodel]
                    # activations = einops.rearrange(activations, 'batch head object dmodel -> batch object head dmodel')
                    activations = activations[:, layer[2], :, :] # Shape [batch object dmodel]
                elif layer[1] == 'attn_unprojected_output_per_head':
                    is_attention=True
                    layer_num = int(layer[0].split('_')[1][0])
                    head_num = layer[2]
                    activations = cache[layer[0]][layer[1]] # Shape [batch object head dmodel]
                    activations = activations[:, :, layer[2], :] # Shape [batch object dmodel]
                else:
                    raise NotImplementedError
            else:
                if layer == 'object_net':
                    activations = cache[layer]['output'] # shape [batch object dmodel]
                    is_attention=False
                else:
                    raise NotImplementedError
            # include_for_regression_mask = (torch.isin(types, torch.Tensor(target_key_objects))).to(device) # Shape [batch object]
            mask = ((torch.isin(x[...,-1], torch.Tensor(true_incl))) & 
                    (torch.isin(types, torch.Tensor(target_query_objects)))
                    ).to(device) # shape [batch object]
            # flat_include_for_regression_mask = einops.rearrange(include_for_regression_mask, 'batch object -> (batch object)')
            flat_mask = einops.rearrange(mask, 'batch object -> (batch object)')
            flat_truths = einops.rearrange(x[...,-1], 'batch object -> (batch object)')
            flat_activs = einops.rearrange(activations, 'batch object dmodel -> (batch object) dmodel')

            # Get potentially useful variables for regressing the extracted features using synmbolic regression
            # Call the shape 'object_query' to mean the object which we are looking at the activation for, and object_key for any variable which might be useful in interpreting the activation
            # For now, we will just take the object itself and the object to which it is paying most attention (if this is attention) or nothing (if this is not attention)
            if is_attention:
                # Will take itself, and the other object with most attention paid, as inputs for symbolic regression
                attn = cache[layer[0]]['attn_weights_per_head'][:,layer[2]].detach() # Shape [batch seq_query seq_key]
                attn_masked = attn * (~(torch.eye(types.shape[-1]).to(bool))).unsqueeze(0) # Mask itself, in case it's paying attention to itself (narcissistic mfs)
                best_obj_idx_other_than_self = attn_masked.argmax(-1) # Shape [batch seq_query]
                self_idx = einops.repeat(torch.arange(types.shape[-1]),'object -> batch object',batch=types.shape[0]) # Shape [batch seq_query]
                self_xs = torch.gather(x, 1, einops.repeat(self_idx, 'batch object -> batch object variable', variable=x.shape[-1]))
                other_xs = torch.gather(x, 1, einops.repeat(best_obj_idx_other_than_self, 'batch object -> batch object variable', variable=x.shape[-1]))
                if n_inputs==5:
                    potential_useful = torch.stack( 
                        Get_PtEtaPhiM_fromXYZT(self_xs[...,0], self_xs[...,1], self_xs[...,2], self_xs[...,3], use_torch=True) + (self_xs[...,4],) +
                        Get_PtEtaPhiM_fromXYZT(other_xs[...,0], other_xs[...,1], other_xs[...,2], other_xs[...,3], use_torch=True)  + (other_xs[...,4],),
                        dim=-1
                    ) # Should have shape [batch object_query 2*numvars] where numvars=5 here
                elif n_inputs==4:
                    potential_useful = torch.stack( 
                        Get_PtEtaPhiM_fromXYZT(self_xs[...,0], self_xs[...,1], self_xs[...,2], self_xs[...,3], use_torch=True) + 
                        Get_PtEtaPhiM_fromXYZT(other_xs[...,0], other_xs[...,1], other_xs[...,2], other_xs[...,3], use_torch=True),
                        dim=-1
                    ) # Should have shape [batch 2 numvars] where numvars=5 here
                else:
                    assert(False)
            else:
                # Will take only itself as inputs for symbolic regression
                self_idx = einops.repeat(torch.arange(types.shape[-1]),'object_query -> batch object_query',batch=types.shape[0]) # Shape [batch seq_query]
                self_xs = torch.gather(x, 1, einops.repeat(self_idx, 'batch object_query -> batch object_query variable', variable=x.shape[-1]))
                if n_inputs==5:
                    potential_useful = torch.stack( 
                            Get_PtEtaPhiM_fromXYZT(self_xs[...,0], self_xs[...,1], self_xs[...,2], self_xs[...,3], use_torch=True) + (self_xs[...,4],),
                            dim=-1
                        ) # Should have shape [batch object_query numvars] where numvars=5 here
                elif n_inputs==4:
                    potential_useful = torch.stack( 
                            Get_PtEtaPhiM_fromXYZT(self_xs[...,0], self_xs[...,1], self_xs[...,2], self_xs[...,3], use_torch=True),
                            dim=-1
                        ) # Should have shape [batch object_query numvars] where numvars=5 here
                else:
                    assert(False)
            
            flat_potential_useful = einops.rearrange(potential_useful, 'batch object_query nvar -> (batch object_query) nvar') # After this line, shape [batch*object_query nvar] where nvar is (2 or 1)*(4 or 5) depending on whether it's attention or not and whether the tag info was included

            masked_flat_activs = flat_activs[flat_mask] # shape [(batch object) dmodel]
            masked_flat_truths = flat_truths[flat_mask] # shape [(batch object) 1]
            masked_flat_potential_useful = flat_potential_useful[flat_mask] # shape [(batch object) nvar]
            num_new_activs = min(N_TRAIN-num_activs, len(masked_flat_activs))
            if num_new_activs>0:
                print(f"Adding {num_new_activs} new activations")
                if (layer_activations is None):
                    layer_activations = torch.empty((N_TRAIN, masked_flat_activs.shape[1]))
                    layer_truths = torch.empty((N_TRAIN,))
                    layer_potential_useful = torch.empty((N_TRAIN, masked_flat_potential_useful.shape[1]))
                layer_activations[num_activs:num_activs+num_new_activs] = masked_flat_activs[:num_new_activs]
                layer_truths[num_activs:num_activs+num_new_activs] = masked_flat_truths[:num_new_activs]
                layer_potential_useful[num_activs:num_activs+num_new_activs] = masked_flat_potential_useful[:num_new_activs]
                num_activs += num_new_activs
            if num_activs>=N_TRAIN:
                break
        assert(num_activs==N_TRAIN), "Finished looping through dataloader but not enough activations found!"
        
        results[layer] = run_sae_analysis(layer_activations, layer, layer_potential_useful, layer_truths, hidden_dim=hidden_dim_sae)
        
    
    # Interpret some interesting features
    for layer, layer_result in results.items():
        autoencoder = layer_result['autoencoder']
        input_dim = autoencoder.encoder.weight.shape[1]
        
        # Look at a few features (e.g., those with highest activation)
        feature_activations = layer_result['features'].mean(dim=0)
        top_features = torch.argsort(feature_activations, descending=True)[:5]
        
        for feature_idx in top_features:
            interpret_features(autoencoder, feature_idx.item(), input_dim)
    
    return results

































# From claude
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import einops

def learned_feature_probe(model, 
                           data_loader, 
                           n_inputs, 
                           object_type_mask,
                           head=1,
                           object_type_mask_for_query=True,
                           probe_target='type', 
                           probe_layer='object_net',
                           test_size=0.2, 
                           random_state=42,
                           verbose=1,
                           num_epochs=1000,
                           learning_rate=1e-4,
                           restrict_to_single_ljet_events=False,
                           ):
    """
    Perform feature probing by training a linear probe on hidden representations.
    
    Args:
        model: The neural network model
        data_loader: DataLoader containing the samples
        n_inputs: Number of input features
        object_type_mask: List of which object types we're looking at if it's non-attention layer we're probing, which query object we're looking at if it's attention layer
        head: The head that we're looking at, if attention (None means we average)
        object_type_mask_for_query: True if we want to mask looking for this type of object as the *query*, False if we want to mask looking for this type of object as the *key*
        probe_target: What to probe for. Options:
            - 'type': Probes object type classification
            - 'pt': Probes transverse momentum
            - 'energy': Probes object energy
            - 'eta': Probes pseudorapidity of query object
            - 'delta-eta-with-lepton': Probes pseudorapidity difference between query and lepton
            - 'delta-phi-with-lepton': Probes azimuthal angle difference between query and lepton
            - 'delta-R-with-lepton': Probes angular separation between query and lepton
            - 'delta-eta-with-neutrino': Probes pseudorapidity difference between query and neutrino
            - 'delta-phi-with-neutrino': Probes azimuthal angle difference between query and neutrino
            - 'delta-R-with-neutrino': Probes angular separation between query and neutrino
            # - 'delta-eta-with-ljet': Probes pseudorapidity difference between query and Pt-leading ljet
            # - 'delta-phi-with-ljet': Probes azimuthal angle difference between query Pt-leading ljet
            # - 'delta-R-with-ljet': Probes angular separation between query Pt-leading ljet
            # - 'delta-Rs-with-ljet': Probes squared angular separation between query Pt-leading ljet
        probe_layer: Which layer to probe. Options:
            - 'object_net': Initial object processing layer
            - 'block_X_attention': Specific attention block output
            - 'block_X_post_attention': Post-attention MLP output
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed results. 0 is no printing, 1 is print just the output results, 3 is print training results too
        num_epochs: Number of training epochs for probe
        restrict_to_single_ljet_events: Whether or not we want to restrict to events with a single large-R jet in them (should find a better way to do this)
    
    Returns:
        Dict containing probe performance metrics
    """
    # Reset and extract activations
    assert(('attention' in probe_layer) or (object_type_mask_for_query))
    data_loader._reset_indices()
    batch = next(data_loader)
    x, _, _, types, _, _, _, labels, truth_inclusion = batch.values()
    x, types, labels = x.to(next(model.parameters()).device), types.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
    
    # Extract activations
    cache = extract_all_activations(model, x[...,:n_inputs], types)
    
    # Select the layer to probe
    if probe_layer == 'object_net':
        features = cache['object_net']['output']
    elif 'block_' in probe_layer and '_attention' in probe_layer:
        block_num = int(probe_layer.split('_')[1])
        if head is not None:
            # Take just the relevant head
            features = cache[f'block_{block_num}_attention']['attn_output_per_head'][:,head,...]
        else:
            # Average across heads if multi-head
            features = cache[f'block_{block_num}_attention']['attn_output_per_head'].mean(dim=1)
    elif 'block_' in probe_layer and '_post_attention' in probe_layer:
        block_num = int(probe_layer.split('_')[1])
        features = cache[f'block_{block_num}_post_attention']['output']
    elif 'residuals' in probe_layer:
        residual_point = int(probe_layer.split('_')[1])
        features = get_residual_stream(cache)[residual_point]
    else:
        raise ValueError(f"Unsupported probe_layer: {probe_layer}")
    
    # Prepare the target variable for probing
    if probe_target == 'type':
        # Probe object type classification
        target = types.cpu().numpy()
        probe_net = nn.Sequential(
            nn.Linear(features.shape[-1], len(torch.unique(types))),
            nn.LogSoftmax(dim=-1)
        )
        criterion = nn.NLLLoss()
        probe_type = 'classification'
    elif probe_target == 'pt':
        # Probe transverse momentum
        target = torch.sqrt(x[..., 0]**2 + x[..., 1]**2).cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'energy':
        # Probe object energy
        target = x[..., 3].cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'eta':
        # Probe pseudorapidity
        px, py, pz = x[..., 0], x[..., 1], x[..., 2]
        eta = 0.5 * torch.log((torch.sqrt(px**2 + py**2 + pz**2) + pz) / 
                               (torch.sqrt(px**2 + py**2 + pz**2) - pz))
        target = eta.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-eta-with-lepton':
        _, Eta, _, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        lepEta = Eta[torch.arange(len(types)), ((types==0).to(int) + (types==1).to(int)).argmax(dim=-1)]
        deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object') - einops.rearrange(lepEta, 'batch -> batch 1'))
        deltaEtas = deltaEtasOG.abs() # Shape [batch objectquery objectkey]
        
        target = deltaEtas.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-phi-with-lepton':
        _, _, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        lepPhi = phi[torch.arange(len(types)), ((types==0).to(int) + (types==1).to(int)).argmax(dim=-1)]
        deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object') - einops.rearrange(lepPhi, 'batch -> batch 1'))
        deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi) # Shape [batch objectquery objectkey]
        
        target = deltaPhis.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-R-with-lepton':
        _, Eta, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        lepEta = Eta[torch.arange(len(types)), ((types==0).to(int) + (types==1).to(int)).argmax(dim=-1)]
        deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object') - einops.rearrange(lepEta, 'batch -> batch 1'))
        deltaEtas = deltaEtasOG.abs() # Shape [batch objectquery objectkey]
        lepPhi = phi[torch.arange(len(types)), ((types==0).to(int) + (types==1).to(int)).argmax(dim=-1)]
        deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object') - einops.rearrange(lepPhi, 'batch -> batch 1'))
        deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi) # Shape [batch objectquery objectkey]

        deltaRs = (deltaPhis**2 + deltaEtas**2).sqrt()
        
        target = deltaRs.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-eta-with-neutrino':
        _, Eta, _, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        neutrinoEta = Eta[torch.arange(len(types)), (types==2).to(int).argmax(dim=-1)]
        deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object') - einops.rearrange(neutrinoEta, 'batch -> batch 1'))
        deltaEtas = deltaEtasOG.abs() # Shape [batch objectquery objectkey]
        
        target = deltaEtas.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-phi-with-neutrino':
        _, _, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        neutrinoPhi = phi[torch.arange(len(types)), (types==2).to(int).argmax(dim=-1)]
        deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object') - einops.rearrange(neutrinoPhi, 'batch -> batch 1'))
        deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi) # Shape [batch objectquery objectkey]
        
        target = deltaPhis.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-R-with-neutrino':
        _, Eta, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        neutrinoEta = Eta[torch.arange(len(types)), (types==2).to(int).argmax(dim=-1)]
        deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object') - einops.rearrange(neutrinoEta, 'batch -> batch 1'))
        deltaEtas = deltaEtasOG.abs() # Shape [batch objectquery objectkey]
        neutrinoPhi = phi[torch.arange(len(types)), (types==2).to(int).argmax(dim=-1)]
        deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object') - einops.rearrange(neutrinoPhi, 'batch -> batch 1'))
        deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi) # Shape [batch objectquery objectkey]

        deltaRs = (deltaPhis**2 + deltaEtas**2).sqrt()
        
        target = deltaRs.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-eta-with-largeRjet':
        _, Eta, _, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        LjetEta = Eta[torch.arange(len(types)), (types==3).to(int).argmax(dim=-1)]
        deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object') - einops.rearrange(LjetEta, 'batch -> batch 1'))
        deltaEtas = deltaEtasOG.abs() # Shape [batch objectquery objectkey]
        
        target = deltaEtas.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-phi-with-largeRjet':
        _, _, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        LjetPhi = phi[torch.arange(len(types)), (types==3).to(int).argmax(dim=-1)]
        deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object') - einops.rearrange(LjetPhi, 'batch -> batch 1'))
        deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi) # Shape [batch objectquery objectkey]
        
        target = deltaPhis.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-R-with-largeRjet':
        _, Eta, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        LjetEta = Eta[torch.arange(len(types)), (types==3).to(int).argmax(dim=-1)]
        deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object') - einops.rearrange(LjetEta, 'batch -> batch 1'))
        deltaEtas = deltaEtasOG.abs() # Shape [batch objectquery objectkey]
        LjetPhi = phi[torch.arange(len(types)), (types==3).to(int).argmax(dim=-1)]
        deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object') - einops.rearrange(LjetPhi, 'batch -> batch 1'))
        deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi) # Shape [batch objectquery objectkey]

        deltaRs = (deltaPhis**2 + deltaEtas**2).sqrt()
        
        target = deltaRs.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    elif probe_target == 'delta-Rs-with-largeRjet':
        _, Eta, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
        LjetEta = Eta[torch.arange(len(types)), (types==3).to(int).argmax(dim=-1)]
        deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object') - einops.rearrange(LjetEta, 'batch -> batch 1'))
        deltaEtas = deltaEtasOG.abs() # Shape [batch objectquery objectkey]
        LjetPhi = phi[torch.arange(len(types)), (types==3).to(int).argmax(dim=-1)]
        deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object') - einops.rearrange(LjetPhi, 'batch -> batch 1'))
        deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi) # Shape [batch objectquery objectkey]

        deltaRs = (deltaPhis**2 + deltaEtas**2)
        
        target = deltaRs.cpu().numpy()
        probe_net = nn.Linear(features.shape[-1], 1)
        criterion = nn.MSELoss()
        probe_type = 'regression'
    else:
        raise ValueError(f"Unsupported probe_target: {probe_target}")
    
    # Get mask for which event-object locations to include
    object_mask = torch.isin(types, torch.tensor(object_type_mask))
    if restrict_to_single_ljet_events: # Only consider events where there's exactly one lepton, for testing purposes
        # for _ in range(3):
        print("WARNING: You're only looking at events with exactly one large-R jet (presumably for testing) - you should re-write this so that we average over all jets, or select a single one based on Pt, or something...")
        object_mask = object_mask & einops.repeat((types==3).sum(dim=-1)==1, 'batch-> batch object', object=types.shape[-1])

    # Flatten features and move to numpy
    features_flat = features.detach().cpu().numpy().reshape(-1, features.shape[-1])
    inclusion_mask_flat = object_mask.detach().cpu().numpy().reshape(features_flat.shape[0], 1) # hopefully this is the right shape [batch*object 1]
    target_flat = target.reshape(-1)

    features_flat = features_flat[inclusion_mask_flat.squeeze()]
    target_flat = target_flat[inclusion_mask_flat.squeeze()]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_flat, target_flat, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    
    if probe_type == 'classification':
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
    else:
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Train the probe
    optimizer = optim.Adam(probe_net.parameters(), lr=learning_rate)
    
    if verbose > 0:
        print(f"Training with {len(y_train)} training samples, {len(y_test)} test samples")
    for epoch in range(num_epochs): 
        # Forward pass
        outputs = probe_net(X_train)
        
        # Compute loss
        if probe_type == 'classification':
            loss = criterion(outputs, y_train)
        else:
            loss = criterion(outputs, y_train)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_every=100
        # if (verbose > 2) and ((epoch%print_every)==(print_every-1)):
        if (verbose > 2) and ((epoch%print_every)==0):
            print(f"Probe for {probe_target} at layer {probe_layer}, epoch {epoch:5d}/{num_epochs:5d}, loss: {loss.item():.3e}")
    
    # Evaluate the probe
    probe_net.eval()
    with torch.no_grad():
        test_outputs = probe_net(X_test)
        train_outputs = probe_net(X_train)
        
        if probe_type == 'classification':
            # Classification metrics
            _, predicted = torch.max(test_outputs, 1)
            _, predicted_train = torch.max(train_outputs, 1)
            accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
            accuracy_train = accuracy_score(y_train.numpy(), predicted_train.numpy())
            metrics = {
                'accuracy': accuracy,
                'accuracy_train': accuracy_train,
                'probe_type': probe_type
            }
        else:
            # Regression metrics
            mse = mean_squared_error(y_test.numpy(), test_outputs.numpy())
            mse_train = mean_squared_error(y_train.numpy(), train_outputs.numpy())
            metrics = {
                'mse': mse,
                'mse_train': mse_train,
                'probe_type': probe_type
            }
    
    if verbose > 0:
        print(f"Probing {probe_target} at layer {probe_layer}" + f" Head {head}"*(head is not None))
        if probe_type == 'classification':
            print(f"Probe Accuracy: {metrics['accuracy']:.4f} (train {metrics['accuracy_train']:.4f})")
        else:
            print(f"Probe MSE: {metrics['mse']:.4f} (train {metrics['mse_train']:.4f})", end='')
            print(f" [Assuming flat prediction: {mean_squared_error(y_test.numpy(), np.ones_like(y_test.numpy())*y_test.numpy().mean())} ({mean_squared_error(y_train.numpy(), np.ones_like(y_train.numpy())*y_train.numpy().mean())})]")
    
    return metrics

# Example usage:
# metrics_type = learned_feature_probe(model, data_loader, n_inputs, 
#                                      probe_target='type', 
#                                      probe_layer='object_net')
# metrics_pt = learned_feature_probe(model, data_loader, n_inputs, 
#                                    probe_target='pt', 
#                                    probe_layer='block_1_attention')







import torch
import torch.nn as nn
import numpy as np
import einops
import matplotlib.pyplot as plt
import seaborn as sns

def track_information_flow(model, data_loader, n_inputs, target_object_types=None):
    """
    Track how information from specific object types propagates through the network.
    
    Args:
        model: The neural network model
        data_loader: DataLoader containing the samples
        n_inputs: Number of input features
        target_object_types: List of object types to track (e.g., [0, 1, 2])
                              If None, track all object types
    
    Returns:
        Dict containing information flow metrics
    """
    # Reset and extract activations
    data_loader._reset_indices()
    batch = next(data_loader)
    x, _, _, types, _, _, _, labels, truth_inclusion = batch.values()
    x, types = x.to(next(model.parameters()).device), types.to(next(model.parameters()).device)
    
    # Extract activations
    cache = extract_all_activations(model, x[...,:n_inputs], types)
    
    # If no specific types specified, use all unique types
    if target_object_types is None:
        target_object_types = torch.unique(types).cpu().numpy()
    
    # Information flow tracking
    information_flow = {}
    
    # Track initial embeddings
    type_embeddings = model.type_embedding(types)
    
    # Track information flow through each attention block
    for i in range(model.num_attention_blocks):
        block_flow = {}
        
        # Get attention weights and outputs
        attn_weights = cache[f"block_{i}_attention"]["attn_weights"]
        attn_output = cache[f"block_{i}_attention"]["attn_output_per_head"]
        
        # Analyze for each target object type
        for obj_type in target_object_types:
            # Create mask for specific object type
            type_mask = (types == obj_type)
            
            # Analyze attention patterns
            type_attn_weights = attn_weights[type_mask]
            type_attn_output = attn_output[type_mask]
            
            # Compute average attention across heads
            avg_attn_weights = type_attn_weights.mean(dim=0)
            
            # Compute information propagation metrics
            block_flow[obj_type] = {
                'avg_attention_weights': avg_attn_weights.cpu().numpy(),
                'attention_output_variance': type_attn_output.var(dim=(0,1)).cpu().numpy()
            }
        
        information_flow[f"block_{i}"] = block_flow
    
    return information_flow

def visualize_information_flow(information_flow, save_path=None):
    """
    Visualize information flow and attention patterns.
    
    Args:
        information_flow: Dict from track_information_flow
        save_path: Optional path to save visualization
    """
    # Number of blocks and object types
    n_blocks = len([k for k in information_flow.keys() if k.startswith('block_')])
    n_types = len(list(information_flow['block_0'].keys()))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_blocks, n_types, figsize=(15, 10))
    
    # Iterate through blocks and object types
    for block_idx in range(n_blocks):
        block_key = f"block_{block_idx}"
        for type_idx, obj_type in enumerate(information_flow[block_key].keys()):
            # Get attention weights
            attn_weights = information_flow[block_key][obj_type]['avg_attention_weights']
            
            # Plot heatmap
            sns.heatmap(attn_weights, 
                        ax=axes[block_idx, type_idx],
                        cmap='viridis', 
                        cbar=False)
            
            axes[block_idx, type_idx].set_title(f"Block {block_idx}, Type {obj_type}")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def activation_maximization_for_attention_heads(model, initial_input, object_types, target_query_index, target_key_index, target_head=None):
    """
    Perform activation maximization to understand what attention heads 'look for'.
    
    Args:
        model: The neural network model. This will remain the same
        initial_input: The initial input (sensible-ish looking x of [1 object variables]). This is what we'll change
        initial_input: The initial types (sensible-ish looking x of [1 object]). This will remain the same
        target_query_index: The index along the object dimension of the object we're interested in maximising the query power of
        target_key_index: The index along the object dimension of the object we're interested in maximising the key response power of
        object_query: 
        target_head: Specific head to analyze (if None, analyze all heads)
    
    Returns:
        Maximized inputs that activate the specified attention heads
    """
    def compute_activation_score(inputs, types, object_query, object_key):
        """
        Compute an activation score for the target attention heads.
        """
        # Zero gradients
        inputs.requires_grad = True
        
        # Forward pass
        # types = torch.full((inputs.shape[0],), target_object_type, device=inputs.device)
        model.zero_grad()
        
        # Extract activations during forward pass
        cache = extract_all_activations(model, inputs, types)
        
        # Compute activation score
        if target_head is not None:
            # Specific head activation
            block_attn = cache['block_0_attention']['attn_output_per_head']
            # activation_score = block_attn[:, target_head, :, :].sum()
            activation_score = block_attn[:, target_head, object_query, object_key]
        else:
            # Average across all heads
            block_attn = cache['block_0_attention']['attn_output_per_head']
            # activation_score = block_attn.sum()
            activation_score = block_attn[:, target_head, object_query, object_key]
        
        # Compute gradients
        print(f"Activation Score: {activation_score.item():.3e}")
        activation_score.backward()
        
        return inputs.grad
    
    # Initialize random inputs
    # inputs = torch.randn(10, n_inputs).to(next(model.parameters()).device)
    inputs = initial_input
    types = object_types
    inputs.requires_grad = True
    types.requires_grad = False
    
    # Gradient ascent
    learning_rate = 0.0001
    iterations = 100
    
    for _ in range(iterations):
        # Compute gradients
        grad = compute_activation_score(inputs, types, target_query_index, target_key_index)
        
        # Update inputs
        inputs = inputs + learning_rate * grad
        
        # Normalize inputs
        inputs = inputs / inputs.norm(dim=1, keepdim=True)
    
    return inputs.detach()


# classifier_weights = model.classifier[0].weight[class_idx]  # shape: [batch, obj, hidden_dim]
# TODO Repeat but instead of maximising the activation, maximise the output *in the direction of some logit* by multiplying by classifier_weights
def activation_maximization_for_attention_heads_KeepFixed(model, initial_input, object_types, target_query_index, target_key_index, target_layer, maximise_direction=None, target_head=None, lr=1e-3, iters=1000, print_every=100):
    """
    Perform activation maximization to understand what attention heads 'look for'. 
    Keeps all but the key object fixed
    
    Args:
        model: The neural network model. This will remain the same
        initial_input: The initial input (sensible-ish looking x of [1 object variables]). This is what we'll change
        object_types: The initial types (sensible-ish looking x of [1 object]). This will remain the same
        target_query_index: The index along the object dimension of the object we're interested in maximising the query power of
        target_key_index: The index along the object dimension of the object we're interested in maximising the key response power of
        maximise_direction: the logit direction in which to maximise. If None, maximise the attention weight (for attention heads). If a single integer, maximise the increase of this logit (0 is 'this particle is in neither', 1 is 'This particle is in SM Higgs', 2 is 'This particle is in W boson'). If a tuple pair of integers (i, j), it's 'maximise the logit diff from class i to class j'
        target_head: Specific head to analyze (if None, analyze all heads)
    
    Returns:
        Maximized inputs that activate the specified attention heads
    """
    def compute_activation_score(inputs, types, object_query, object_key, maximise_direction=None, verbose=False, just_activ=False):
        """
        Compute an activation score for the target attention heads.
        """
        # Zero gradients
        # inputs.requires_grad = True
        # inputs.grad_fn.grad_mode = True
        
        # Forward pass
        # types = torch.full((inputs.shape[0],), target_object_type, device=inputs.device)
        model.zero_grad()

        
        # Extract activations during forward pass
        cache = extract_all_activations(model, inputs, types, detach=False)
        
        # Compute activation score
        if target_head is not None:
            # Specific head activation
            # block_attn = cache['block_0_attention']['attn_output_per_head']
            # activation_score = block_attn[:, target_head, :, :].sum()
            
            if maximise_direction is None:
                block_attn_wts = cache[f'block_{target_layer}_attention']['attn_weights_per_head']
                activation_score = block_attn_wts[:, target_head, object_query, object_key]
            else:
                block_attn_wts = cache[f'block_{target_layer}_attention']['attn_output_per_head'] # shape [batch head object_query dmodel]
                activation_score = einops.einsum(block_attn_wts[:, target_head, object_query], maximise_direction, 'batch d_model, d_model -> batch')
        else:
            raise NotImplementedError
            # block_attn = cache[f'block_{target_layer}_attention']['attn_output_per_head']
            # Average across all heads
            block_attn = cache[f'block_{target_layer}_attention']['attn_weights_per_head']
            # activation_score = block_attn.sum()
            activation_score = block_attn[:, target_head, object_query, object_key]
        if just_activ:
            return activation_score.item()
        # Compute gradients
        if verbose:
            print(f"Activation Score: {activation_score.item():.3e}")
        activation_score.backward()
        
        return inputs.grad, activation_score.item()
    
    if maximise_direction is not None:
        if isinstance(maximise_direction, int):
            maximise_direction = model.classifier[0].weight[maximise_direction]
        elif isinstance(maximise_direction, tuple):
            assert(len(maximise_direction)==2)
            maximise_direction = model.classifier[0].weight[maximise_direction[1]] - model.classifier[0].weight[maximise_direction[0]]

    # Initialize random inputs
    # inputs = torch.randn(10, n_inputs).to(next(model.parameters()).device)
    if 1:
        inputs = torch.tensor(initial_input.data, requires_grad=True)
        orig_input = torch.tensor(initial_input.data, requires_grad=False)
        types = torch.tensor(object_types.data, requires_grad=False)
    else:
        inputs = initial_input
        types = object_types
    object_mask = torch.zeros_like(initial_input, dtype=torch.bool)
    object_mask[:, target_key_index, :] = 1
    
    # Gradient ascent
    scores=[]
    for iter in range(iters):
        # Compute gradients
        if (iter%print_every)==0:
            print(f"Step: {iter}/{iters}, ", end='')
        grad, score = compute_activation_score(inputs, types, target_query_index, target_key_index, maximise_direction, verbose=((iter%print_every)==0))
        scores.append(score)
        
        # Zero out gradients for non-target objects
        grad = torch.where(object_mask, grad, torch.zeros_like(grad))
        
        # Update only the target object
        # with torch.no_grad():
            # inputs = inputs + learning_rate * grad
        inputs = inputs + lr * grad.detach()
        
        
        # Normalize inputs]
        # inputs = inputs / inputs.norm(dim=1, keepdim=True)
        # inputs = inputs / inputs.norm(dim=(1,2), keepdim=True)
        # Reset the input so we can do a clean grad on it next time. This seems kinda hacky, 
        # we have to do it because otherwise we get a warning on the second iteration of this 
        # loop telling us that "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed"
        inputs = torch.tensor(inputs.data, requires_grad=True)
    
    # Change ONLY this variable (in Pt, Eta, Phi, M, tag)-space
    altered_scores = []
    # for variable in range(inputs.shape[-1]-1, -1, -1):
    for variable in range(inputs.shape[-1]):
        new_input = torch.Tensor(orig_input.clone().detach().cpu().numpy())
        ptEtaPhiM_New = list(Get_PtEtaPhiM_fromXYZT(*(new_input[:, target_key_index, var] for var in range(4)), use_torch=True))
        ptEtaPhiM_Maxed = list(Get_PtEtaPhiM_fromXYZT(*(inputs[:, target_key_index, var] for var in range(4)), use_torch=True))
        if variable<4:
            ptEtaPhiM_New[variable]=ptEtaPhiM_Maxed[variable]

        xyztNew = list(GetXYZT_FromPtEtaPhiM(*(np.array(i.item()).reshape(1,) for i in ptEtaPhiM_New)))
        for var in range(4):
            new_input[...,target_key_index,var]=xyztNew[var].item()

        if variable == 4:
            new_input[:, target_key_index, variable] = inputs[:, target_key_index, variable]
        altered_scores.append(compute_activation_score(new_input, types, target_query_index, target_key_index, maximise_direction, verbose=False, just_activ=True)) # Only care about the actual score here

    # Change all BUT this variable (in Pt, Eta, Phi, M, tag)-space    
    unaltered_scores = []
    # for variable in range(inputs.shape[-1]-1, -1, -1):# for variable in range(inputs.shape[-1]-1, -1, -1):
    for variable in range(inputs.shape[-1]):
        new_input = torch.Tensor(orig_input.clone().detach().cpu().numpy())
        ptEtaPhiM_New = list(Get_PtEtaPhiM_fromXYZT(*(new_input[:, target_key_index, var] for var in range(4)), use_torch=True))
        ptEtaPhiM_Maxed = list(Get_PtEtaPhiM_fromXYZT(*(inputs[:, target_key_index, var] for var in range(4)), use_torch=True))
        for var in range(4):
            if var != variable:
                ptEtaPhiM_New[var]=ptEtaPhiM_Maxed[var]

        xyztNew = list(GetXYZT_FromPtEtaPhiM(*(np.array(i.item()).reshape(1,) for i in ptEtaPhiM_New)))
        for var in range(4):
            new_input[...,target_key_index,var]=xyztNew[var].item()

        if variable != 4:
            new_input[:, target_key_index, variable] = inputs[:, target_key_index, variable]
        unaltered_scores.append(compute_activation_score(new_input, types, target_query_index, target_key_index, maximise_direction, verbose=False, just_activ=True)) # Only care about the actual score here
    # print("WARNING: The unaltered/altered scores are in terms of px,py,pz,pt,tag rather than pt,eta,phi,M,tag, so not easy to compare.")
    # print("         Should really re-do this calculation in order to use these properly")
    return inputs.detach(), scores, altered_scores, unaltered_scores

# Example usage:
# 1. Track information flow
# flow_metrics = track_information_flow(model, data_loader, n_inputs)
# visualize_information_flow(flow_metrics, save_path='information_flow.png')

# 2. Activation maximization
# maximized_inputs = activation_maximization_for_attention_heads(
#     model, 
#     target_object_type=2,  # Neutrino
#     target_head=0  # First attention head
# )
































import torch

def get_intervention_hook(cache: ActivationCache, name: str, intervention_fn=None, detach=True):
    """
    Create a hook function that applies an intervention to the activations.
    """
    def hook_fn(module, input, kwargs, output):
        # Apply the intervention function if specified
        if detach:
            cache.store[f"{name}_unmodified"] = [x.detach() if isinstance(x, torch.Tensor) else x for x in output]
        else:
            cache.store[f"{name}_unmodified"] = output
        # Actually apply the intervention
        if intervention_fn:
            output = intervention_fn(output)
        # Save the modified output to the cache
        if detach:
            cache.store[f"{name}_modified"] = [x.detach() if isinstance(x, torch.Tensor) else x for x in output]
        else:
            cache.store[f"{name}_modified"] = output
        return output  # Ensure the modified output is returned
    return hook_fn

def ablate_attention_head(output, head_idx, num_heads):
    """
    Ablates a specific attention head by zeroing its contribution in the output.
    """
    # Reshape the output to separate heads (assuming [batch_size, seq_len, embed_dim])
    # Ouput is a tuple containing attn_output, attn_output_weights. We only care about the attn_output
    batch_size, seq_len, embed_dim = output[0].shape
    head_dim = embed_dim // num_heads
    reshaped_output = output[0].view(batch_size, seq_len, num_heads, head_dim)
    
    # Zero out the specified head
    reshaped_output[:, :, head_idx, :] = 0
    
    # Reshape back to original dimensions & return with (original) attention weights
    return reshaped_output.view(batch_size, seq_len, embed_dim), output[1]

def run_with_interventions(model, inputs, interventions, detach=True):
    """
    Run the model with specified interventions applied at various layers.
    
    Args:
        model: The PyTorch model.
        inputs: The inputs to the model.
        interventions: A dictionary where keys are layer names and values are intervention functions.
        
    Returns:
        The final model output and a cache of intermediate activations.
    """
    if 0:
        cache = ActivationCache()
        
        # Attach hooks for each intervention
        hooks = []
        for layer_name, intervention_fn in interventions.items():
            layer = dict(model.named_modules())[layer_name]
            hook = layer.register_forward_hook(get_intervention_hook(cache, layer_name, intervention_fn))
            hooks.append(hook)
        
        # Run the model
        
        output = model(*inputs)
        
        # Remove hooks after execution
        for hook in hooks:
            hook.remove()
        
        return output, cache
    else:
        cache = ActivationCache()
        
        # Create hooks for all components
        hooks = [
            (model.object_net, get_activation_hook(cache, "object_net", detach=detach)),
            (model.type_embedding, get_activation_hook(cache, "type_embedding", detach=detach))
        ]
        
        # Add hooks for attention blocks
        hooks.extend(hook_attention_heads(model, cache, detach=detach))
        
        # Attach hooks for each intervention
        for layer_name, intervention_fn in interventions:
            layer = dict(model.named_modules())[layer_name]
            hook = (layer, get_intervention_hook(cache, layer_name, intervention_fn)) # Following run_with_hooks convention, we give the layer first and then the hook function
            hooks.append(hook)
        
        # Run the model with hooks
        output = run_with_hooks(
            model,
            inputs,
            fwd_hooks=hooks
        )
        
        # Add the output to the cache
        if detach:
            cache.store["output"] = output.detach()
        else:
            # print("WARNING: Running cache with non-detached output - not sure how safe this is...")
            cache.store["output"] = output
        
        return output, cache