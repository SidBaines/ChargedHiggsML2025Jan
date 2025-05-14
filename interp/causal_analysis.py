import torch
from torch import nn
from collections import defaultdict
from typing import Dict, List, Tuple, Callable, Any, Optional

from .activations import ModelActivationExtractor, run_with_hooks, ActivationCache # Assuming run_with_hooks is accessible

class ActivationPatcher:
    """
    Performs activation patching (causal tracing) by running the model with
    specified activations replaced by values from a source run or custom values.
    """
    def __init__(self, model: nn.Module, model_activation_extractor: ModelActivationExtractor):
        self.model = model
        self.model_activation_extractor = model_activation_extractor # Used to get source activations

    def _get_module_by_path(self, path_str: str) -> nn.Module:
        """
        Retrieves a module from the model using a dot-separated path string.
        Example: 'attention_blocks.0.self_attention'
        """
        module = self.model
        for part in path_str.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _create_patching_hook(self, patch_value_fn: Callable, 
                              source_cache: Optional[Dict], 
                              target_output_component_idx: Optional[int] = None) -> Callable:
        """
        Creates a forward hook that replaces the output of a module.
        """
        def hook_fn(module, input_args, kwargs, output_val): # output_val is original module output
            # patch_value_fn returns:
            # 1. A single tensor if target_output_component_idx is not None (this tensor replaces one part of output_val).
            # 2. The entire new output (tensor or tuple) if target_output_component_idx is None.
            new_val_from_patch_fn = patch_value_fn(source_cache, output_val)

            if target_output_component_idx is not None and isinstance(output_val, tuple):
                # We are replacing a specific component of the original tuple output_val.
                # new_val_from_patch_fn is expected to be the tensor for that component.
                if not isinstance(new_val_from_patch_fn, torch.Tensor):
                    raise TypeError(
                        f"patch_value_fn was expected to return a Tensor when "
                        f"target_output_component_idx ({target_output_component_idx}) is specified, "
                        f"but got {type(new_val_from_patch_fn)}."
                    )
                output_list = list(output_val)
                output_list[target_output_component_idx] = new_val_from_patch_fn.clone().detach()
                return tuple(output_list)
            else:
                # We are replacing the entire output of the module.
                # new_val_from_patch_fn is the new output, which could be a tensor or a tuple.
                if isinstance(new_val_from_patch_fn, torch.Tensor):
                    return new_val_from_patch_fn.clone().detach()
                elif isinstance(new_val_from_patch_fn, tuple):
                    # If the new output is a tuple, clone/detach its tensor elements
                    return tuple(o.clone().detach() if isinstance(o, torch.Tensor) else o 
                                 for o in new_val_from_patch_fn)
                else:
                    # If it's some other type (e.g., None, or a non-tensor/tuple value), return as is.
                    return new_val_from_patch_fn
        return hook_fn

    def patch_and_run(self,
                      original_inputs_tuple: Tuple[torch.Tensor, ...],
                      patch_operations: List[Dict[str, Any]],
                      source_inputs_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
                      output_fn: Optional[Callable[[torch.Tensor], Any]] = None
                     ) -> Any:
        """
        Runs the model on `original_inputs_tuple` with specified activations patched.

        Args:
            original_inputs_tuple: Tuple of input tensors (e.g., features, types) for the main run.
            patch_operations: A list of dictionaries, each defining a patching operation:
                - 'target_module_path': Dot-separated string path to the nn.Module whose output is patched.
                - 'patch_value_fn': Callable(source_cache, original_module_output) -> tensor_to_patch_with.
                                    `source_cache` is the activation cache from `source_inputs_tuple` (or None).
                                    `original_module_output` is what the target module would have outputted.
                - 'target_output_component_idx': (Optional) If the target module returns a tuple,
                                                 which component to replace (e.g., 0 for MHA's attn_output).
            source_inputs_tuple: (Optional) Tuple of input tensors for the "source" run.
                                 Activations from this run can be used by `patch_value_fn`.
            output_fn: (Optional) A function to process the final model output after patching.
                       If None, returns the raw model output.

        Returns:
            The processed model output (or raw output if output_fn is None).
        """
        source_cache = None
        if source_inputs_tuple is not None:
            # Ensure source_inputs_tuple is correctly unpacked if it's (features, types)
            source_cache = self.model_activation_extractor.extract_activations(source_inputs_tuple)

        fwd_hooks = []
        for op_spec in patch_operations:
            target_module = self._get_module_by_path(op_spec['target_module_path'])
            patch_hook = self._create_patching_hook(
                patch_value_fn=op_spec['patch_value_fn'],
                source_cache=source_cache,
                target_output_component_idx=op_spec.get('target_output_component_idx')
            )
            fwd_hooks.append((target_module, patch_hook))

        # Ensure original_inputs_tuple is correctly unpacked for run_with_hooks
        patched_model_output = run_with_hooks(
            self.model,
            original_inputs_tuple, # Should be (features, types)
            fwd_hooks=fwd_hooks
        )

        if output_fn:
            return output_fn(patched_model_output)
        return patched_model_output


class DirectLogitAttributor:
    """
    Analyzes the contribution of different model components to the final logits
    using ablation (zeroing out component outputs).
    """
    def __init__(self, model: nn.Module, 
                 model_activation_extractor: ModelActivationExtractor,
                 activation_patcher: ActivationPatcher):
        self.model = model
        self.model_activation_extractor = model_activation_extractor
        self.activation_patcher = activation_patcher
        self.model_type = self.model_activation_extractor.model_type # 'reconstruction' or 'classification'

    def _get_module_by_path(self, path_str: str) -> nn.Module: # Duplicated for now, consider utils
        module = self.model
        for part in path_str.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _extract_target_logit(self, model_outputs: torch.Tensor, 
                              target_class_idx: int, 
                              batch_idx: int = 0, # Typically analyze one event at a time or mean later
                              target_object_idx: Optional[int] = None) -> torch.Tensor:
        """
        Extracts the specific logit of interest from the model's output.
        Assumes model_outputs is for a single batch element if batch_idx is used,
        or that operations like .mean(dim=0) will be applied by the user if needed.
        """
        if self.model_type == 'reconstruction':
            if target_object_idx is None:
                raise ValueError("target_object_idx must be specified for 'reconstruction' model type.")
            # model_outputs shape: [batch, object, class]
            return model_outputs[batch_idx, target_object_idx, target_class_idx]
        elif self.model_type == 'classification':
            # model_outputs shape: [batch, class]
            return model_outputs[batch_idx, target_class_idx]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_ablation_patch_ops(self, inputs_tuple: Tuple[torch.Tensor, ...], 
                                components_to_ablate: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generates patch_operations dictionaries for ablating specified components.
        This requires knowledge of the model's structure (e.g., TestNetwork).
        """
        patch_ops_dict = {}
        
        # Run a clean pass to get source_cache, needed for head ablations
        # and to know original output shapes for zeroing.
        # This source_cache is from the *same* inputs_tuple as the ablation run.
        source_cache_for_ablation_helpers = self.model_activation_extractor.extract_activations(inputs_tuple)

        for comp_name_template in components_to_ablate:
            if comp_name_template == "object_net_output":
                patch_ops_dict["object_net_output"] = {
                    'target_module_path': 'object_net',
                    'patch_value_fn': lambda sc, orig_out: torch.zeros_like(orig_out),
                    # target_output_component_idx is None here, as object_net returns a tensor
                }
            # Add more components like type_embedding if desired

            # Attention Blocks
            for i in range(self.model.num_attention_blocks):
                block_module = self.model.attention_blocks[i]
                
                # MHA output ablation
                if comp_name_template == "all_mha_outputs" or comp_name_template == f"block_{i}_mha_output":
                    op_name = f"block_{i}_mha_output"
                    patch_ops_dict[op_name] = {
                        'target_module_path': f'attention_blocks.{i}.self_attention',
                        # patch_value_fn now returns only the new attn_output tensor
                        'patch_value_fn': lambda sc, orig_out_tuple: torch.zeros_like(orig_out_tuple[0]),
                        'target_output_component_idx': 0 # Patching the attn_output part (orig_out_tuple[0])
                    }

                # Individual Head output ablation
                if comp_name_template == "all_attention_heads" or f"block_{i}_head_" in comp_name_template:
                    mha_module = block_module['self_attention']
                    num_heads = mha_module.num_heads
                    for h_idx in range(num_heads):
                        if comp_name_template == "all_attention_heads" or comp_name_template == f"block_{i}_head_{h_idx}_output":
                            op_name = f"block_{i}_head_{h_idx}_output"
                            
                            def create_head_ablation_fn(block_idx_val, head_idx_val):
                                def head_ablation_fn(src_cache, original_mha_output_tuple):
                                    # src_cache is source_cache_for_ablation_helpers
                                    # original_mha_output_tuple is (clean_attn_output, clean_attn_weights)
                                    
                                    # Get clean per-head activations [batch, num_heads, seq_len, d_model_after_mha_out_proj_per_head]
                                    # The last dimension is effectively d_model / num_heads if hidden_dim_attn is None,
                                    # or hidden_dim_attn / num_heads if hidden_dim_attn is not None.
                                    # More accurately, it's the dimension of each head's output *after* its slice of out_proj.weight.
                                    clean_attn_output_per_head = src_cache[f'block_{block_idx_val}_attention']['attn_output_per_head'].clone()
                                    
                                    # Ablate the specific head's contribution
                                    clean_attn_output_per_head[:, head_idx_val, :, :] = 0.0 
                                    
                                    # Sum contributions from all heads
                                    # Result shape: [batch, seq_len, d_model_of_mha_output]
                                    # where d_model_of_mha_output is hidden_dim or hidden_dim_attn
                                    ablated_summed_heads = clean_attn_output_per_head.sum(dim=1) 
                                    
                                    current_mha_module = self._get_module_by_path(f'attention_blocks.{block_idx_val}.self_attention')
                                    
                                    # Add the MHA's output projection bias
                                    if current_mha_module.out_proj.bias is not None:
                                        reconstructed_mha_output = ablated_summed_heads + current_mha_module.out_proj.bias
                                    else:
                                        reconstructed_mha_output = ablated_summed_heads
                                    
                                    # If MHA operated in hidden_dim_attn, apply the block's external output_projection
                                    # if self.model.hidden_dim_attn is not None:
                                    #     # Ensure the 'output_projection' layer exists in the block
                                    #     if 'output_projection' in self.model.attention_blocks[block_idx_val]:
                                    #         output_proj_layer = self.model.attention_blocks[block_idx_val]['output_projection']
                                    #         final_ablated_attn_output = output_proj_layer(reconstructed_mha_output)
                                    #     else:
                                    #         # This case should ideally not happen if hidden_dim_attn is set,
                                    #         # as TestNetwork structure implies output_projection exists.
                                    #         # Handle defensively or raise error.
                                    #         # final_ablated_attn_output = reconstructed_mha_output 
                                    #         raise ValueError("hidden_dim_attn is set but no output_projection layer found in block.")
                                    #         # Consider: warnings.warn("hidden_dim_attn is set but no output_projection layer found in block.")
                                    # else:
                                    #     final_ablated_attn_output = reconstructed_mha_output
                                    final_ablated_attn_output = reconstructed_mha_output            
                                    # Return only the ablated attention output tensor
                                    return final_ablated_attn_output
                                return head_ablation_fn

                            patch_ops_dict[op_name] = {
                                'target_module_path': f'attention_blocks.{i}.self_attention',
                                'patch_value_fn': create_head_ablation_fn(i, h_idx),
                                'target_output_component_idx': 0 # We are replacing the MHA's attn_output (1st element of tuple)
                            }
                
                # MLP block output ablation (if MLP exists)
                if self.model.include_mlp and ('post_attention' in block_module):
                    if comp_name_template == "all_mlp_outputs" or comp_name_template == f"block_{i}_mlp_output":
                        op_name = f"block_{i}_mlp_output"
                        patch_ops_dict[op_name] = {
                            'target_module_path': f'attention_blocks.{i}.post_attention',
                            'patch_value_fn': lambda sc, orig_out: torch.zeros_like(orig_out),
                        }
            
            if comp_name_template == "classifier_pre_logits": # Ablate input to classifier
                 patch_ops_dict["classifier_pre_logits"] = {
                    'target_module_path': 'classifier', # This will ablate the *output* of classifier
                                                        # To ablate input, need to know classifier structure
                                                        # Assuming classifier is nn.Sequential(nn.Linear(...))
                                                        # Patching 'classifier.0' (the Linear layer) might be better
                    'patch_value_fn': lambda sc, orig_out: torch.zeros_like(orig_out),
                }


        return patch_ops_dict

    def analyze_logit_attribution(self,
                                  inputs_tuple: Tuple[torch.Tensor, ...],
                                  target_class_idx: int,
                                  components_to_ablate: List[str], # e.g. ["all_attention_heads", "block_0_mlp_output"]
                                  batch_idx: int = 0,
                                  target_object_idx: Optional[int] = None
                                 ) -> Dict[str, float]:
        """
        Calculates the attribution of specified components to a target logit
        by ablating them and observing the change in the logit.

        Args:
            inputs_tuple: Tuple of input tensors (features, types).
            target_class_idx: Index of the target class for logit analysis.
            components_to_ablate: List of strings identifying components.
                                  Examples: "object_net_output", "all_mha_outputs",
                                  "block_0_head_3_output", "all_mlp_outputs".
            batch_idx: Index of the event in the batch to analyze.
            target_object_idx: (For reconstruction models) Index of the target object.

        Returns:
            A dictionary mapping component names to their logit attribution values.
        """
        # 1. Get clean logit
        clean_model_output = self.model(*inputs_tuple)
        clean_logit = self._extract_target_logit(clean_model_output, target_class_idx, 
                                                 batch_idx, target_object_idx).item()

        # 2. Get patch operations for ablation
        # The source_cache for ablation helpers is implicitly created inside _get_ablation_patch_ops
        # using inputs_tuple.
        ablation_patch_ops_map = self._get_ablation_patch_ops(inputs_tuple, components_to_ablate)
        
        attributions = {}

        # 3. For each component, ablate and calculate attribution
        for op_name, patch_op_spec in ablation_patch_ops_map.items():
            # The patch_value_fn in patch_op_spec might use a source_cache.
            # For ablation, it typically uses source_cache (from inputs_tuple) to get shapes or per-head info.
            ablated_model_output = self.activation_patcher.patch_and_run(
                original_inputs_tuple=inputs_tuple,
                patch_operations=[patch_op_spec],
                source_inputs_tuple=inputs_tuple # Crucial for head ablation to have access to clean per-head outputs
            )
            ablated_logit = self._extract_target_logit(ablated_model_output, target_class_idx,
                                                       batch_idx, target_object_idx).item()
            
            attributions[op_name] = clean_logit - ablated_logit
            
        return attributions 





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
