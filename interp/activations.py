"""
Utilities for extracting and analyzing activations from neural network models.
"""

import torch
from torch import nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable, Any
import math
from utils.utils import Get_PtEtaPhiM_fromXYZT


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
        def hook_fn(module, grad_input, kwargs, grad_output):
            cache.store[name]["grad_input"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_input]
            cache.store[name]["grad_output"] = [x.detach() if isinstance(x, torch.Tensor) and x is not None else x for x in grad_output]
    return hook_fn

def hook_attention_heads(model, cache: ActivationCache, detach=True, SINGLE_ATTENTION=False, min_attention=None, bottleneck_attention_output=None):
    """Add hooks to extract attention patterns and outputs from each head.
    
    Args:
        model: The neural network model
        cache: ActivationCache to store results
        detach: Whether to detach tensors from computation graph
        SINGLE_ATTENTION: Flag to force attention to a single (maximally activating) object
        min_attention: Minimum attention threshold, below which attention is zeroed out
        bottleneck_attention_output: Number of neurons for attention bottleneck, None for no bottleneck
        
    Returns:
        List of hooks
    """
    CHECKS_ON = False
    assert((min_attention is None) or (SINGLE_ATTENTION is True))
    assert((min_attention is not None) or (SINGLE_ATTENTION is False))
    hooks = []
    
    # Hook attention blocks
    for i, block in enumerate(model.attention_blocks):
        # Original MultiheadAttention's forward is complex, so we'll need to hook it specially
        def get_attention_hook(block_idx, detach=True, bottleneck_down=None, bottleneck_up=None):
            def hook_fn(module, inputs, kwargs, output):
                '''
                Here, we actually have to go through the calculations of the attention mechanism oursleves, to 
                obtain the per-head output. There are optional checks (turned off usually) to make sure that we
                are calculating the same output as the original method
                '''
                # Extract q, k, v and attention weights (output is just attn_output)
                # For nn.MultiheadAttention, output is (attn_output, attn_weights)
                attn_output, attn_weights = output
                if detach:
                    cache.store[f"block_{block_idx}_attention"]["input"] = [x.detach() if isinstance(x, torch.Tensor) else x for x in inputs]
                    cache.store[f"block_{block_idx}_attention"]["attn_weights"] = attn_weights.detach() # Can keep these as they are actually fine
                    cache.store[f"block_{block_idx}_attention"]["OLD_attn_output"] = attn_output.detach() # This is averaged over head, so we shouldn't use it since we now have per-head
                else:
                    cache.store[f"block_{block_idx}_attention"]["input"] = [x if isinstance(x, torch.Tensor) else x for x in inputs]
                    cache.store[f"block_{block_idx}_attention"]["attn_weights"] = attn_weights # Can keep these as they are actually fine
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
                
                # Ensure this is self-attention (keys=queries=values)
                if not ((q==k).all() and (q==v).all()): 
                    raise NotImplementedError("Only self-attention is supported")
                
                # Process inputs to get per-head values
                q = k = v = q.transpose(1,0) # After this line, shape is now: [object batch dmodel]
                tgt_len, bsz, embed_dim  = q.shape
                src_len, _, _  = k.shape
                
                # Get separate Q, K, V projections
                q, k, v = torch.nn.functional._in_projection_packed(q, k, v, module.in_proj_weight, module.in_proj_bias) 
                
                # Reshape for multihead and with batch first
                num_heads = module.num_heads
                head_dim = E // num_heads
                q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # [batch*head object_query dhead]
                k = k.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # [batch*head object_key dhead]
                v = v.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # [batch*head object_value dhead]
                
                # Compute attention weights
                _B, _Nt, E = q.shape
                q_scaled = q * math.sqrt(1.0 / float(E))
                
                # Apply padding mask if provided
                if key_padding_mask is not None:
                    key_padding_mask = (
                        key_padding_mask.view(bsz, 1, 1, src_len)
                        .expand(-1, num_heads, -1, -1)
                        .reshape(bsz * num_heads, 1, src_len)
                    )   # [batch*head 1 object]
                    attn_weights_byhand = torch.baddbmm(
                        key_padding_mask, q_scaled, k.transpose(-2, -1)
                    ) # [batch*head object_query object_key]
                else:
                    attn_weights_byhand = torch.bmm(q_scaled, k.transpose(-2, -1))

                # Apply softmax to get attention probabilities
                attn_weights_byhand = torch.nn.functional.softmax(attn_weights_byhand, dim=-1) # [batch*head object_query object_key]
                
                # Implement SINGLE_ATTENTION option if specified
                if SINGLE_ATTENTION:
                    # Mask out attention weights below threshold
                    attn_weights_byhand = torch.where(attn_weights_byhand > min_attention, 
                                                     attn_weights_byhand, 
                                                     torch.zeros_like(attn_weights_byhand))

                # Validation check
                if CHECKS_ON and (not SINGLE_ATTENTION):
                    assert(torch.isclose(attn_weights_byhand.view(bsz, num_heads, tgt_len, src_len)[0,1,:,:], 
                                        attn_weights[0,1,:,:], atol=1e-05).all())

                # Apply attention weights to values
                attn_output_byhand = torch.bmm(attn_weights_byhand, v) # [batch*head object_query dhead]
                attn_output_byhand = (
                    attn_output_byhand.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
                ) # [object_query*batch headnum*dhead]
                
                # Compute per-head outputs with projection
                attn_output_per_head = torch.empty((B, module.num_heads, N, module.out_proj.weight.shape[0])).to(module.out_proj.weight.device)
                for head_n in range(num_heads):
                    W_rows = module.out_proj.weight.transpose(0,1)[head_n*head_dim:(head_n+1)*head_dim] # [d_head dmodel]
                    attnoutput_cols = attn_output_byhand[:, head_n*head_dim:(head_n+1)*head_dim] # [object_query*batch d_head]
                    head_output = torch.matmul(attnoutput_cols, W_rows) # [object_query*batch dmodel]
                    attn_output_per_head[:, head_n, :, :] = head_output.contiguous().view(tgt_len, bsz, -1).transpose(0,1)
                
                # Validation check for non-bottleneck case
                if CHECKS_ON and (not SINGLE_ATTENTION) and (bottleneck_attention_output is None):
                    recrafted_attn_output = (attn_output_per_head.sum(dim=1) + module.out_proj.bias)
                    assert(torch.isclose(recrafted_attn_output[0,:5,0], attn_output[0,:5,0], atol=1e-05).all())
                
                # Apply bottleneck if requested
                if bottleneck_attention_output is not None:
                    bottlneck_activations = torch.empty((B, module.num_heads, N, bottleneck_attention_output)).to(module.out_proj.weight.device)
                    assert ((bottleneck_up is not None) and (bottleneck_down is not None))
                    for head_n in range(num_heads):
                        # Get head-specific output [batch, seq_len, model_dim]
                        head_out = attn_output_per_head[:, head_n, :, :]
                        # Project down [batch*seq_len, model_dim] -> [batch*seq_len, bottleneck_dim]
                        projected_down = torch.matmul(
                            head_out.contiguous().view(-1, head_out.size(-1)), 
                            bottleneck_down[head_n].weight.t()
                        )
                        bottlneck_activations[:, head_n, :, :] = projected_down.view(head_out.size(0), head_out.size(1), -1)
                        
                        # Project back up [batch*seq_len, bottleneck_dim] -> [batch*seq_len, model_dim]
                        projected_up = torch.matmul(
                            projected_down, 
                            bottleneck_up[head_n].weight.t()
                        )
                        # Reshape back to [batch, seq_len, model_dim]
                        attn_output_per_head[:, head_n, :, :] = projected_up.view(head_out.size(0), head_out.size(1), -1)
                
                # Store the processed outputs in the cache
                if detach:
                    cache.store[f"block_{block_idx}_attention"]["attn_weights_per_head"] = attn_weights_byhand.view(bsz, num_heads, tgt_len, src_len).detach()
                    cache.store[f"block_{block_idx}_attention"]["attn_unprojected_output_per_head"] = attn_output_byhand.view(tgt_len, bsz, num_heads, head_dim).transpose(0,1).detach()
                    cache.store[f"block_{block_idx}_attention"]["attn_output_per_head"] = attn_output_per_head.detach()
                    if bottleneck_attention_output is not None:
                        cache.store[f"block_{block_idx}_attention"]["bottleneck_activation"] = bottlneck_activations.detach()
                else:
                    cache.store[f"block_{block_idx}_attention"]["attn_weights_per_head"] = attn_weights_byhand.view(bsz, num_heads, tgt_len, src_len)
                    cache.store[f"block_{block_idx}_attention"]["attn_unprojected_output_per_head"] = attn_output_byhand.view(tgt_len, bsz, num_heads, head_dim).transpose(0,1)
                    cache.store[f"block_{block_idx}_attention"]["attn_output_per_head"] = attn_output_per_head
                    if bottleneck_attention_output is not None:
                        cache.store[f"block_{block_idx}_attention"]["bottleneck_activation"] = bottlneck_activations
                
                # Return either the modified or original output
                if SINGLE_ATTENTION or (bottleneck_attention_output is not None):
                    return (attn_output_per_head.sum(dim=1) + module.out_proj.bias), attn_weights
                else:
                    return output
            return hook_fn

        def patch_attention(m):
            """Patch MultiheadAttention to return attention weights"""
            forward_orig = m.forward
            def wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            m.forward = wrap
            
        # Patch the attention module if needed
        if str(type(block['self_attention'].forward)) == "<class 'method'>":
            # hasn't been patched yet
            patch_attention(block['self_attention'])
        elif str(type(block['self_attention'].forward)) == "<class 'function'>":
            # Has been patched already
            pass
        else:
            assert(False) # Shouldn't get here
            
        # Add hook with proper bottleneck parameters if needed
        if bottleneck_attention_output is not None:
            hooks.append((block['self_attention'], get_attention_hook(i, detach=detach, bottleneck_down=block.bottleneck_down, bottleneck_up=block.bottleneck_up)))
        else:
            hooks.append((block['self_attention'], get_attention_hook(i, detach=detach)))

        # Hook the layernorm and post-attention module if needed
        if model.include_mlp:
            hooks.append((block['post_attention'], get_activation_hook(cache, f"block_{i}_post_attention", detach=detach)))
    
    
    return hooks




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

def extract_all_activations(model, object_features, object_types, single_attention=False, min_attention=None, detach=True, interventions=None, verbose=False):
    """
    Extract and return all important intermediate activations from the model.
    
    Args:
        model: The neural network model
        object_features: Tensor of object features [batch, objects, features]
        object_types: Tensor of object types [batch, objects]
        detach: Whether to detach tensors from computation graph
        
    Returns:
        ActivationCache containing all activations
    """
    cache = ActivationCache()
    
    # Create hooks for initial components
    hooks = [
        (model.object_net, get_activation_hook(cache, "object_net", detach=detach)),
        (model.type_embedding, get_activation_hook(cache, "type_embedding", detach=detach))
    ]
    # And interventions for the initial components
    if interventions is not None:
        for layer_name, intervention_fn in interventions:
            if layer_name.startswith("object_net"):
                if verbose:
                    print(f"Adding intervention to object_net")
                hook_fn = get_intervention_hook(cache, layer_name, intervention_fn, detach)
                hooks.append((model.object_net, hook_fn))
            elif layer_name.startswith("type_embedding"):
                if verbose:
                    print(f"Adding intervention to type_embedding")
                hook_fn = get_intervention_hook(cache, layer_name, intervention_fn, detach)
                hooks.append((model.type_embedding, hook_fn))
    
    # Add hooks for attention blocks
    # This includes bottleneck attention if it exists and optional MLP layers
    hooks.extend(hook_attention_heads(model, cache, detach=detach, 
                                     SINGLE_ATTENTION=single_attention, 
                                     min_attention=min_attention,
                                     bottleneck_attention_output=model.bottleneck_attention))
    
    # Intervention hooks for attention blocks
    if interventions is not None:
        for i, block in enumerate(model.attention_blocks):
            for layer_name, intervention_fn in interventions:
                # Find the appropriate module based on layer name
                if layer_name == f"attention_blocks.{i}.self_attention":
                    if verbose:
                        print(f"Adding intervention to block {i} attention")
                    module = block.self_attention
                    hook_fn = get_intervention_hook(cache, layer_name, intervention_fn, detach)
                    # Replace the standard hook with the intervention hook.
                    # Or should we be appending it, since we still want to extract the unmodified activations?
                    # I think we should be appending it, since we still want to extract the unmodified activations.
                    # But we should be careful not to double-count the intervention.

                    # for j, (mod, _) in enumerate(fwd_hooks):
                    #     if mod == module:
                    #         fwd_hooks[j] = (module, hook_fn)
                    #         break
                    # else:
                    #     fwd_hooks.append((module, hook_fn))
                    hooks.append((module, hook_fn))
    
    # Hook the final classifier
    hooks.append((model.classifier, get_activation_hook(cache, "classifier", detach=detach)))
    # And interventions for the final classifier
    if interventions is not None:
        for layer_name, intervention_fn in interventions:
            if layer_name.startswith("classifier"):
                if verbose:
                    print(f"Adding intervention to classifier")
                hook_fn = get_intervention_hook(cache, layer_name, intervention_fn, detach)
                hooks.append((model.classifier, hook_fn))
    
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
        cache.store["output"] = output
    
    return cache

def get_residual_stream(cache: ActivationCache, verbose=False):
    """
    Extract the additions to each residual stream from each attention block.
    
    Args:
        cache: ActivationCache with model activations
        
    Returns:
        Dict mapping block index to residual stream tensor
    """
    residuals = {}
    
    # First residual is the output of the object_net
    residuals["object_net"] = cache["object_net"]["output"]
    prev_residual = residuals["object_net"]
    
    # Extract additions to the residual stream from each attention block
    for block_num, attention_block in enumerate([i for i in cache.store.keys() if (('block_' in i) and (not('post' in i)))]):
        if verbose:
            print(f"Adding attention block {attention_block} attention output to residual stream dict")
        for h_idx in range(cache[attention_block]['attn_output_per_head'].shape[1]):
            residuals[f'block_{block_num}_head_{h_idx}'] = prev_residual + cache[attention_block]['attn_output_per_head'][:, h_idx, :, :]
            prev_residual = residuals[f'block_{block_num}_head_{h_idx}']
    # Extract additions to the residual stream from each post-attention MLP
    for i in range(len([i for i in cache.store.keys() if 'post_attention' in i])):
        if f"block_{i}_post_attention" in cache.store:
            if verbose:
                print(f"Adding post-attention MLP {i} output to residual stream dict")
            residuals[f'block_{i}_post_attention'] = prev_residual + cache[f"block_{i}_post_attention"]["output"]
            prev_residual = residuals[f'block_{i}_post_attention']
    
    return residuals

class ModelActivationExtractor:
    """Class to extract and organize activations from model runs.
    
    This class provides a simplified interface for extracting activations
    from different model architectures, supporting both reconstruction and
    classification tasks.
    """
    def __init__(self, model, model_type='reconstruction'):
        """
        Args:
            model: The neural network model
            model_type: Either 'reconstruction' [batch, object, class] or 
                        'classification' [batch, class]
        """
        self.model = model
        self.model_type = model_type
        self.cache = ActivationCache()
        
    def extract_activations(self, inputs, targets=None, subset_indices=None, 
                           single_attention=False, min_attention=None, interventions=None):
        """Extract activations from the model for given inputs.
        
        Args:
            inputs: Tuple of (features, types) tensors
            targets: Optional target values for filtering
            subset_indices: Optional indices for analyzing a subset of data
            single_attention: Whether to force attention to single objects
            min_attention: Minimum attention threshold
            interventions: Optional dictionary of interventions to apply to the model.
        Returns:
            ActivationCache with extracted activations
        """
        # Apply subset selection if provided
        if subset_indices is not None:
            features, types = inputs
            if isinstance(subset_indices, tuple) and len(subset_indices) == 2:
                # Subset on both batch and object dimensions
                batch_idx, obj_idx = subset_indices
                features = features[batch_idx][:, obj_idx]
                types = types[batch_idx][:, obj_idx]
            else:
                # Subset on batch dimension only
                features = features[subset_indices]
                types = types[subset_indices]
            inputs = (features, types)
        
        # Extract activations using existing function
        self.cache = extract_all_activations(
            self.model, 
            inputs[0],  # features
            inputs[1],  # types
            single_attention=single_attention,
            min_attention=min_attention,
            interventions=interventions
        )
        
        return self.cache
        
    def get_residual_stream(self):
        """Get the residual stream from the cache."""
        return get_residual_stream(self.cache)
    
    def get_model_outputs(self, inputs, subset_indices=None):
        """Get model outputs with appropriate shapes based on model type.
        
        Args:
            inputs: Tuple of (features, types) tensors
            subset_indices: Optional indices for analyzing a subset of data
            
        Returns:
            Model outputs with proper shape
        """
        features, types = inputs
        if subset_indices is not None:
            if isinstance(subset_indices, tuple) and len(subset_indices) == 2:
                batch_idx, obj_idx = subset_indices
                features = features[batch_idx][:, obj_idx]
                types = types[batch_idx][:, obj_idx]
            else:
                features = features[subset_indices]
                types = types[subset_indices]
        
        outputs = self.model(features, types)
        
        # Handle different output shapes based on model type
        if self.model_type == 'reconstruction':
            # Output shape: [batch, object, class]
            return outputs
        elif self.model_type == 'classification':
            # Output shape: [batch, class]
            return outputs
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def extract_activations_filtered(self, 
                                   dataloader,
                                   filter_condition: Callable,
                                   max_samples: int = 1000,
                                   single_attention: bool = False,
                                   min_attention: float = None,
                                   interventions: Optional[Dict[str, Callable]] = None):
        """Extract activations from filtered data.
        
        Args:
            dataloader: Data iterator
            filter_condition: Function to select data
            max_samples: Maximum samples to process
            single_attention: Whether to force attention to single objects
            min_attention: Minimum attention threshold
            
        Returns:
            ActivationCache with extracted activations from filtered data
        """
        from .data_filtering import DataFilter
        
        # Get the padding token from the model
        padding_token = self.model.num_particle_types - 1
        
        # Create data filter
        data_filter = DataFilter(padding_token=padding_token)
        
        # Filter the events
        filtered_batch = data_filter.filter_events(
            dataloader,
            filter_condition,
            max_events=max_samples
        )
        
        # Extract features and types
        features = filtered_batch.get('features', filtered_batch.get('x'))
        types = filtered_batch.get('types', filtered_batch.get('object_types'))
        
        # Extract activations
        self.cache = extract_all_activations(
            self.model,
            features,
            types,
            single_attention=single_attention,
            min_attention=min_attention,
            interventions=interventions
        )
        
        return self.cache 

class DirectLogitContributionAnalyzer:
    """
    Analyzes the direct contribution of model components' activations 
    (as they appear in the residual stream) to the final logits.
    This is done by projecting the component's residual stream vector 
    through the final classifier layer.
    """
    def __init__(self, model: nn.Module, 
                 model_activation_extractor: ModelActivationExtractor):
        self.model = model
        self.model_activation_extractor = model_activation_extractor
        self.model_type = self.model_activation_extractor.model_type # 'reconstruction' or 'classification'

    def _get_module_by_path(self, path_str: str) -> nn.Module: # Duplicated for now
        module = self.model
        for part in path_str.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _get_final_classifier_params(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieves the weights and bias of the final classifier linear layer.
        Assumes the classifier is model.classifier and the linear layer is the first element
        if it's a Sequential module, or the module itself if it's a Linear layer.
        """
        if isinstance(self.model.classifier, nn.Linear):
            final_linear_layer = self.model.classifier
        elif isinstance(self.model.classifier, nn.Sequential) and \
             len(self.model.classifier) > 0 and \
             isinstance(self.model.classifier[0], nn.Linear):
            final_linear_layer = self.model.classifier[0]
        else:
            raise ValueError("Could not identify the final linear classifier layer. "
                             "Expected model.classifier to be nn.Linear or nn.Sequential(nn.Linear, ...)")
        
        return final_linear_layer.weight, final_linear_layer.bias

    def _get_component_residual_stream_activation(self,
                                                  component_name: str,
                                                  block_idx: Optional[int],
                                                  head_idx: Optional[int],
                                                  full_clean_cache: ActivationCache) -> torch.Tensor:
        """
        Extracts or computes the activation of a component as it would appear in 
        the main residual stream (i.e., with dimension model.hidden_dim).

        Args:
            component_name: Type of component (e.g., "object_net", "mha", "head", "mlp").
            block_idx: Index of the attention block, if applicable.
            head_idx: Index of the attention head, if applicable.
            full_clean_cache: ActivationCache from a clean run.

        Returns:
            Tensor of shape [batch, seq_len, hidden_dim] representing the component's
            contribution to the residual stream.
        """
        if component_name == "object_net_output":
            return full_clean_cache['object_net']['output']

        # Ensure block_idx is provided for block-specific components
        if block_idx is None and component_name in ["mha_output", "head_output", "mlp_output"]:
            raise ValueError(f"block_idx must be provided for component_name '{component_name}'")

        if component_name == "mha_output":
            # Output of MHA block (after MHA's out_proj and block's output_projection if any)
            # This is already in hidden_dim.
            return full_clean_cache[f'block_{block_idx}_attention']['output'][0]
        
        if component_name == "mlp_output":
            if not self.model.include_mlp or f'block_{block_idx}_mlp' not in full_clean_cache.store:
                raise ValueError(f"MLP output requested for block {block_idx}, but MLP might be disabled or hook missing.")
            return full_clean_cache[f'block_{block_idx}_mlp']['output']

        if component_name == "head_output":
            if head_idx is None:
                raise ValueError("head_idx must be provided for component_name 'head_output'")

            # Get per-head activations after their slice of MHA's W_O
            # Shape: [batch, num_heads, seq_len, d_slice]
            # where d_slice = (hidden_dim_attn or hidden_dim) / num_heads
            attn_output_per_head = full_clean_cache[f'block_{block_idx}_attention']['attn_output_per_head']
            
            # Select the specific head's output
            # Shape: [batch, seq_len, d_slice]
            z_h = attn_output_per_head[:, head_idx, :, :]
            
            _b, _s, d_slice = z_h.shape
            
            # The vector z_h needs to be conceptually placed into the full residual stream dimension.
            # If hidden_dim_attn is used, it's first placed into hidden_dim_attn, then projected.
            # If not, it's placed into hidden_dim.
            
            if self.model.hidden_dim_attn is not None:
                # Path: z_h -> (embed into hidden_dim_attn) -> block_output_projection -> hidden_dim
                target_dim_before_block_proj = self.model.hidden_dim_attn
                
                # Create a zero tensor of shape [B, S, hidden_dim_attn]
                x = torch.zeros(_b, _s, target_dim_before_block_proj, device=z_h.device, dtype=z_h.dtype)
                
                # Place z_h into its slice
                start_idx = head_idx * d_slice
                x[:, :, start_idx : start_idx + d_slice] = z_h
                
                # Apply the block's output projection
                block_output_proj_layer = self.model.attention_blocks[block_idx].get('output_projection')
                if block_output_proj_layer is None:
                    raise ValueError(f"hidden_dim_attn is set, but block {block_idx} has no 'output_projection' layer.")
                return block_output_proj_layer(x) # Shape: [B, S, hidden_dim]
            else:
                # Path: z_h -> (embed into hidden_dim) -> hidden_dim (no further block projection for this path)
                target_dim_hidden = self.model.hidden_dim # Classifier expects this
                
                # Create a zero tensor of shape [B, S, hidden_dim]
                x = torch.zeros(_b, _s, target_dim_hidden, device=z_h.device, dtype=z_h.dtype)
                
                # Place z_h into its slice
                start_idx = head_idx * d_slice
                x[:, :, start_idx : start_idx + d_slice] = z_h
                return x # Shape: [B, S, hidden_dim]
                
        raise ValueError(f"Unknown component_name for direct contribution analysis: {component_name}")


    def analyze_contribution(self,
                             inputs_tuple: Tuple[torch.Tensor, ...],
                             target_class_idx: int,
                             components_to_analyze: List[Dict[str, Any]], # e.g. [{'name': 'head_output', 'block': 0, 'head': 1}]
                             batch_idx: int = 0,
                             target_object_idx: Optional[int] = None,
                             include_bias_in_contribution: bool = True
                            ) -> Dict[str, float]:
        """
        Calculates the direct contribution of specified components to a target logit.

        Args:
            inputs_tuple: Tuple of input tensors (features, types).
            target_class_idx: Index of the target class for logit analysis.
            components_to_analyze: List of dicts, each specifying a component.
                Each dict must have 'name' (e.g., "object_net_output", "mha_output", "head_output", "mlp_output").
                For "mha_output", "head_output", "mlp_output", add 'block': block_idx.
                For "head_output", also add 'head': head_idx.
            batch_idx: Index of the event in the batch to analyze.
            target_object_idx: (For reconstruction models) Index of the target object.
                               (For classification models with per-object components) If provided, analyze this object's
                               contribution to the (potentially pooled) logit. If None, and component is per-object,
                               its contribution is averaged over objects before projection.
            include_bias_in_contribution: Whether to include the final classifier bias term
                                          apportioned to this component's contribution.
                                          If True, adds W_final @ act + B_final[target_class].
                                          If False, adds W_final @ act.
                                          Note: Apportioning bias is non-trivial. A common simplification
                                          is to analyze W_final @ act and consider bias separately.
                                          For simplicity, if True, we add the full target class bias.

        Returns:
            A dictionary mapping descriptive component names to their direct logit contribution values.
        """
        # 1. Get clean activations
        full_clean_cache = self.model_activation_extractor.extract_activations(inputs_tuple)

        # 2. Get final classifier weights and bias
        W_final, B_final = self._get_final_classifier_params() # W_final: [num_classes, hidden_dim]

        contributions = {}

        for comp_spec in components_to_analyze:
            comp_type_name = comp_spec['name']
            block_idx = comp_spec.get('block')
            head_idx = comp_spec.get('head')

            descriptive_name = comp_type_name
            if block_idx is not None:
                descriptive_name += f"_block{block_idx}"
            if head_idx is not None:
                descriptive_name += f"_head{head_idx}"

            # Get the component's activation in the residual stream [batch, seq_len, hidden_dim]
            comp_residual_activation = self._get_component_residual_stream_activation(
                comp_type_name, block_idx, head_idx, full_clean_cache
            )

            # Select the specific activation vector for projection
            # comp_resid_act_single_event is [seq_len, hidden_dim] or [hidden_dim]
            comp_resid_act_single_event = comp_residual_activation[batch_idx] 

            act_to_project = None # Should be [hidden_dim]

            if self.model_type == 'reconstruction':
                if target_object_idx is None:
                    raise ValueError("target_object_idx must be specified for 'reconstruction' model type.")
                if comp_resid_act_single_event.ndim == 2: # [seq_len, hidden_dim]
                    act_to_project = comp_resid_act_single_event[target_object_idx, :]
                elif comp_resid_act_single_event.ndim == 1: # Already [hidden_dim], e.g. post-pooling
                     act_to_project = comp_resid_act_single_event
                else:
                    raise ValueError(f"Unexpected activation dimension for reconstruction: {comp_resid_act_single_event.ndim}")

            elif self.model_type == 'classification':
                if comp_resid_act_single_event.ndim == 2: # [seq_len, hidden_dim] (pre-pooling component)
                    if target_object_idx is not None:
                        # Contribution of a specific object's activation (before pooling) to the final logit
                        act_to_project = comp_resid_act_single_event[target_object_idx, :]
                    else:
                        # Average contribution across objects
                        act_to_project = comp_resid_act_single_event.mean(dim=0) 
                elif comp_resid_act_single_event.ndim == 1: # [hidden_dim] (already pooled or CLS token like)
                    act_to_project = comp_resid_act_single_event
                else:
                    raise ValueError(f"Unexpected activation dimension for classification: {comp_resid_act_single_event.ndim}")
            
            if act_to_project is None:
                 raise ValueError("Could not determine activation vector for projection.")


            # Project onto the target class direction
            # W_final[target_class_idx, :] is [hidden_dim]
            logit_val = torch.einsum('d,d->', W_final[target_class_idx, :], act_to_project)

            if include_bias_in_contribution and B_final is not None:
                logit_val = logit_val + B_final[target_class_idx]
            
            contributions[descriptive_name] = logit_val.item()
            
        return contributions 

def extract_filtered_activations(
    dataloader,
    model,
    n_features: int,
    event_filter_fn=None,
    object_filter_fn_query=None,
    object_filter_fn_key=None,
    activations_to_return=None,
    interventions=None,
    max_samples=1000,
    device=None,
    verbose=False,
    PtEtaPhiM_format=True,
):
    """
    Run the model on the dataloader, collect activations, and filter by event/object selection criteria.

    Args:
        dataloader: DataLoader yielding batches of data.
        model: The neural network model.
        event_filter_fn: Function(batch_dict) -> mask or indices for event selection.
        object_filter_fn_query: Function(batch_dict, types, ...) -> mask or indices for query object selection.
        object_filter_fn_key: Function(batch_dict, types, ...) -> mask or indices for key object selection (optional).
        activations_to_return: List of cache keys to extract (e.g. ['block_0_attention.bottleneck_activation']).
        interventions: Optional interventions to apply.
        max_samples: Maximum number of events to process.
        device: Device to run the model on.
        verbose: Print progress.

    Returns:
        Dictionary with:
            - 'activations': dict of requested activations, each [N, ...]
            - 'query_features': [N, ...]
            - 'key_features': [N, ...] (if key selection is used)
            - 'event_indices': [N]
            - 'object_indices': [N]
    """
    from collections import defaultdict

    model.eval()
    if device is not None:
        model = model.to(device)

    all_acts = defaultdict(list)
    all_query_features = []
    all_key_features = []
    all_event_indices = []
    all_object_indices = []

    n_seen = 0
    for batch_idx, batch in enumerate(dataloader):
        if n_seen >= max_samples:
            break

        # Move to device if needed
        features = batch['x'][..., :n_features]
        types = batch['types']
        if device is not None:
            features = features.to(device)
            types = types.to(device)

        # Run model and extract activations
        cache = extract_all_activations(
            model, features, types, interventions=interventions, detach=True
        )

        batch_size = features.shape[0]
        n_objects = features.shape[1]

        # Event selection
        if event_filter_fn is not None:
            event_mask = event_filter_fn(batch)
            if torch.is_tensor(event_mask):
                event_mask = event_mask.cpu().numpy()
            event_indices = [i for i in range(batch_size) if event_mask[i]]
        else:
            event_indices = list(range(batch_size))

        # For each selected event, select objects
        for i in event_indices:
            # Query object selection
            if object_filter_fn_query is not None:
                query_mask = object_filter_fn_query(batch, types[i], i) # Shape: [n_objects]
                if torch.is_tensor(query_mask):
                    query_mask = query_mask.cpu().numpy()
                query_indices = [j for j in range(n_objects) if query_mask[j]]
            else:
                query_indices = list(range(n_objects))

            # For each query object, extract activations and features
            for j in query_indices:
                # Optionally, for each key object (e.g. for attention-based selection)
                if object_filter_fn_key is not None:
                    for act_key in activations_to_return:
                        
                        # Key object selection (optional)
                        # We're not actually selecting a key object here; rather we are selecting
                        #   along the query dimension, but we are doing it based on whether or not 
                        #   the key object that the query object is paying attention to passes some
                        #   given criteria
                        # Find out which object is being paid more than 90% of the attention 
                        # (otherwise we can't hope to meaningfully reconstruct the activation 
                        # from the inputs...)
                        block, subkey, head = act_key.split('.')
                        head = int(head)
                        assert (('attention' in block) and (not('post' in block))) # Make sure we're looking at an attention block
                        attn = cache[block]['attn_weights_per_head'][i, head, j, :] # Shape: [key_object]
                        key_mask = object_filter_fn_key(batch, types[i], i) # Shape: [n_objects]
                        # Check that something is being paid more than 90% of the attention and if so passes the 
                        # key filter
                        key_mask = key_mask & (attn > 0.9) # Shape: [n_objects]
                        if not torch.any(key_mask):
                            continue
                        # There can only be one key object that passes the filter
                        assert torch.sum(key_mask) == 1
                        # Get the index of the key object
                        key_index = torch.where(key_mask)[0][0]
                        act = cache[block][subkey][i, head, j, :]  # [bottleneck_dim]
                        all_acts[act_key].append(act.cpu()) # Shape: [bottleneck_dim]
                        if PtEtaPhiM_format:
                            all_query_features.append(
                                torch.concatenate((
                                    torch.tensor(Get_PtEtaPhiM_fromXYZT(*[features[i, j][v] for v in range(4)], use_torch=True)),
                                    torch.tensor([features[i, j][4:n_features]])
                                    ),
                                    dim=0
                                ).cpu()
                            )
                            all_key_features.append(
                                torch.concatenate((
                                    torch.tensor(Get_PtEtaPhiM_fromXYZT(*[features[i, key_index][v] for v in range(4)], use_torch=True)),
                                    torch.tensor([features[i, key_index][4:n_features]])
                                    ),
                                    dim=0
                                ).cpu()
                                )
                        else:
                            all_query_features.append(features[i, j].cpu())
                            all_key_features.append(features[i, key_index].cpu())
                        all_event_indices.append(i + batch_idx * batch_size)
                        all_object_indices.append(j)
                else:
                    for act_key in activations_to_return:
                        block, subkey, head = act_key.split('.')
                        head = int(head)
                        act = cache[block][subkey][i, head, j, :]  # [bottleneck_dim]
                        all_acts[act_key].append(act.cpu())
                        if PtEtaPhiM_format:
                            all_query_features.append(
                                torch.concatenate((
                                    torch.tensor(Get_PtEtaPhiM_fromXYZT(*[features[i, j][v] for v in range(4)], use_torch=True)),
                                    torch.tensor([features[i, j][4:n_features]])
                                    ),
                                    dim=0
                                ).cpu()
                            )
                        else:
                            all_query_features.append(features[i, j].cpu())
                    all_event_indices.append(i + batch_idx * batch_size)
                    all_object_indices.append(j)

                n_seen += 1
                if n_seen >= max_samples:
                    break
            if n_seen >= max_samples:
                break
        if n_seen >= max_samples:
            break

    # Stack results
    result = {
        'activations': {k: torch.stack(v) for k, v in all_acts.items()},
        'query_features': torch.stack(all_query_features),
        'event_indices': all_event_indices,
        'object_indices': all_object_indices,
    }
    if all_key_features:
        result['key_features'] = torch.stack(all_key_features)
    return result 