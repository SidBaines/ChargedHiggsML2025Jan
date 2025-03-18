import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
from utils import Get_PtEtaPhiM_fromXYZT
import einops
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler

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
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Register backward hooks
        if bwd_hooks:
            for module, hook_fn in bwd_hooks:
                hooks.append(module.register_backward_hook(hook_fn))
        
        # Run the model
        output = model(*inputs)
        
        return output
    
    finally:
        # Clean up by removing hooks
        if clear_hooks:
            for hook in hooks:
                hook.remove()

def get_activation_hook(cache: ActivationCache, name: str):
    """Create a hook function that saves activations to the cache."""
    def hook_fn(module, input, output):
        cache.store[name]["input"] = [x.detach() if isinstance(x, torch.Tensor) else x for x in input]
        cache.store[name]["output"] = output.detach()
    return hook_fn

def get_gradient_hook(cache: ActivationCache, name: str):
    """Create a hook function that saves gradients to the cache."""
    def hook_fn(module, grad_input, grad_output):
        cache.store[name]["grad_input"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_input]
        cache.store[name]["grad_output"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_output]
    return hook_fn

def hook_attention_heads(model, cache: ActivationCache):
    """Add hooks to extract attention patterns and outputs from each head."""
    hooks = []
    
    # Hook attention blocks
    for i, block in enumerate(model.attention_blocks):
        # Original MultiheadAttention's forward is complex, so we'll need to hook it specially
        def get_attention_hook(block_idx):

            if ATTN_METHOD_1:
                def hook_fn(module, inputs, output):
                    # The output from MultiheadAttention is already (attn_output, attn_weights)
                    # We just need to capture the weights before they're averaged
                    # Get the raw attention weights from the module's _get_attention_weights method
                    with torch.no_grad():
                        q, k, v = inputs[0], inputs[1], inputs[2]
                        B, N, E = q.shape
                        scaling = float(E) ** -0.5
                        
                        # Calculate attention scores
                        q = q.view(B, N, module.num_heads, E // module.num_heads)
                        k = k.view(B, N, module.num_heads, E // module.num_heads)
                        
                        # Calculate attention weights
                        attn = torch.einsum('bnhe,bmhe->bhnm', q, k) * scaling
                        
                        # Apply mask if present
                        if len(inputs) > 3 and isinstance(inputs[3], dict):
                            key_padding_mask = inputs[3].get('key_padding_mask')
                            if key_padding_mask is not None:
                                attn = attn.masked_fill(
                                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),
                                )
                        
                        # Apply softmax
                        attn = torch.softmax(attn, dim=-1)
                    
                    # Store the per-head attention weights
                    cache.store[f"block_{block_idx}_attention"]["attn_weights"] = attn.detach()
                    attn_output, _ = output
                    cache.store[f"block_{block_idx}_attention"]["attn_output"] = attn_output.detach()
                    # Return original output unchanged
                    return output
            else:
                    
                def hook_fn(module, inputs, output):
                    # Extract q, k, v and attention weights (output is just attn_output)
                    # For nn.MultiheadAttention, output is (attn_output, attn_weights)
                    attn_output, attn_weights = output
                    cache.store[f"block_{block_idx}_attention"]["attn_weights"] = attn_weights.detach()
                    cache.store[f"block_{block_idx}_attention"]["attn_output"] = attn_output.detach()
                    return output
            return hook_fn


        def patch_attention(m):
            forward_orig = m.forward
            def wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            m.forward = wrap
        patch_attention(block['self_attention'])

        hooks.append((block['self_attention'], get_attention_hook(i)))


        
        # Hook the layernorm and post-attention module
        # hooks.append((block['layer_norm'], get_activation_hook(cache, f"block_{i}_layernorm")))
        hooks.append((block['post_attention'], get_activation_hook(cache, f"block_{i}_post_attention")))
    
    # Hook the final classifier
    hooks.append((model.classifier, get_activation_hook(cache, "classifier")))
    
    return hooks

def extract_all_activations(model, object_features, object_types):
    """
    Extract and return all important intermediate activations from the model.
    """
    cache = ActivationCache()
    
    # Create hooks for all components
    hooks = [
        (model.object_net, get_activation_hook(cache, "object_net")),
        (model.type_embedding, get_activation_hook(cache, "type_embedding"))
    ]
    
    # Add hooks for attention blocks
    hooks.extend(hook_attention_heads(model, cache))
    
    # Run the model with hooks
    output = run_with_hooks(
        model,
        (object_features, object_types),
        fwd_hooks=hooks
    )
    
    # Add the output to the cache
    cache.store["output"] = output.detach()
    
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
            residuals[i+1] = cache[f"block_{i}_post_attention"]["output"]
    
    return residuals






import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import seaborn as sns

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
        attn_weights = cache[f"block_{i}_attention"]["attn_weights"]
        attn_output = cache[f"block_{i}_attention"]["attn_output"]
        
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

def analyze_object_type_attention(model, cache, object_types, padding_token, ret_dict=False, combine_elec_and_muon=False):
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
                    
                    # Apply masks and compute average attention
                    masked_attn = head_attn * query_mask * key_mask
                    
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
def get_direct_logit_attribution(model, 
                                 cache,
                                 truth_inclusion,
                                #  pred_inclusion, # Can get this from logits
                                 object_types, 
                                 true_incl:list,
                                 type_incl:list,
                                 class_idx,
                                 include_mlp=True
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

    # Compute attributions for each attention block
    for i in range(model.num_attention_blocks):
        # Get attention outputs
        attn_weights = cache[f"block_{i}_attention"]["attn_weights"]
        attn_output = cache[f"block_{i}_attention"]["attn_output"]
        mlp_output = cache[f"block_{i}_post_attention"]['output']
        
        
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

        # Get MLP contributions
        attributions[f"block_{i}_mlp"] = einops.einsum(mlp_output, classifier_weights, 'batch object dmod, batch object dmod -> batch object')
    
    # Aggregate attributions across batches for analysis
    aggregated = {}
    for key, attr in attributions.items():
        # Average across batch and objects
        aggregated[key] = einops.einsum(attr* mask, '... batch object -> ...') / einops.einsum(mask, 'batch object ->')
        # aggregated[key] = attr.mean(dim=(1, 2)).cpu().numpy()
    
    
    conts = np.zeros((model.num_attention_blocks, num_heads + include_mlp))
    for i in range(model.num_attention_blocks):
        conts[i,:num_heads] = aggregated[f"block_{i}_attention"].detach().cpu().numpy()
        if include_mlp:
            conts[i,-1] = aggregated[f"block_{i}_mlp"].detach().cpu()

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
                                    include_mlp=include_mlp
        )
        plt.subplot(1,3,class_idx+1)
        vmax=max(vmax, abs(r[class_idx]).max())
    for class_idx in range(3):
        plt.subplot(1,3,class_idx+1)
        if TRANSPOSE:
            plt.imshow(r[class_idx].transpose(), cmap='PiYG', vmin=-vmax, vmax=vmax)
        else:
            plt.imshow(r[class_idx], cmap='PiYG', vmin=-vmax, vmax=vmax)
        plt.title(f"Class {class_idx} DLA")
        if TRANSPOSE and (class_idx==2):
            plt.colorbar(fraction=0.046, pad=0.04)
        if (not TRANSPOSE) and (class_idx==2):
            plt.colorbar(fraction=0.066, pad=0.04)
        if (not TRANSPOSE) and include_mlp:
            plt.xticks(range(3), ['Head 0', 'Head 1', 'MLP'], rotation=85)
    if title is None:
        plt.suptitle(f"Logit attribution for true incl. {true_incl}, type {type_incl}")
    else:
        plt.suptitle(title)
    if not TRANSPOSE:
        fig.supxlabel(f"Posn in layer")
        fig.supylabel(f"Layer #")
    plt.tight_layout()
    return fig