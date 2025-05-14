import torch
from typing import Callable, Dict, Tuple
from tqdm.auto import tqdm
from utils.utils import Get_PtEtaPhiM_fromXYZT  # Import your existing physics utilities

class DataFilter:
    """Advanced filtering system for physics data.
    
    This class provides methods to filter data at both the event level and object level
    based on complex conditions, including spatial relationships between objects.
    """
    
    def __init__(self, padding_token: int, device: str = "cpu"):
        """
        Args:
            padding_token: Token ID that represents padding
            device: Device to perform computations on
        """
        self.padding_token = padding_token
        self.device = device
        
    def filter_events(self, 
                     dataloader, 
                     condition_fn: Callable, 
                     max_events: int = 1000,
                     return_indices: bool = False) -> Dict:
        """Filter events based on a condition function.
        
        Args:
            dataloader: Data iterator
            condition_fn: Function that takes batch data and returns boolean mask
                          of shape [batch_size]
            max_events: Maximum number of events to collect
            return_indices: Whether to return original indices of selected events
            
        Returns:
            Dictionary of filtered batch data
        """
        filtered_batches = []
        event_indices = []
        total_collected = 0
        batch_offset = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Filtering events")):
                # Apply condition function to get boolean mask for events
                event_mask = condition_fn(batch)
                
                if event_mask.any():
                    # Filter all tensors in the batch
                    filtered_batch = {
                        k: v[event_mask] if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    filtered_batches.append(filtered_batch)
                    
                    # Track original indices if requested
                    if return_indices:
                        original_indices = torch.where(event_mask)[0] + batch_offset
                        event_indices.append(original_indices)
                    
                    # Update count
                    total_collected += event_mask.sum().item()
                    
                if total_collected >= max_events:
                    break
                    
                batch_offset += len(batch[next(iter(batch.keys()))])
        
        # Combine all filtered batches
        combined_batch = {}
        if filtered_batches:
            for key in filtered_batches[0].keys():
                if isinstance(filtered_batches[0][key], torch.Tensor):
                    combined_batch[key] = torch.cat([batch[key] for batch in filtered_batches])
                else:
                    # Handle non-tensor data if present
                    combined_batch[key] = [item for batch in filtered_batches for item in batch[key]]
        
        if return_indices:
            combined_batch['original_indices'] = torch.cat(event_indices) if event_indices else torch.tensor([], device=self.device)
            
        return combined_batch
    
    def filter_objects(self, 
                      batch_data: Dict, 
                      condition_fn: Callable,
                      preserve_event_structure: bool = False) -> Tuple[Dict, torch.Tensor]:
        """Filter objects within events based on a condition function.
        
        Args:
            batch_data: Dictionary of batch tensors
            condition_fn: Function that takes batch data and returns boolean mask
                          of shape [batch_size, max_objects]
            preserve_event_structure: If True, keep the original event structure
                                     by setting non-selected objects to padding
                                     
        Returns:
            Tuple of (filtered_batch, object_mask)
        """
        # Apply condition to get object mask
        object_mask = condition_fn(batch_data)
        
        if preserve_event_structure:
            # Create a new batch with the same structure but with non-selected objects padded
            filtered_batch = {k: v.clone() for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
            
            # Get features and types tensors
            features = filtered_batch.get('features', filtered_batch.get('x'))
            types = filtered_batch.get('types', filtered_batch.get('object_types'))
            
            # Set non-selected objects to padding
            types_mask = ~object_mask & (types != self.padding_token)
            types[types_mask] = self.padding_token
            
            # Optionally zero out features of padded objects
            if features is not None:
                features[types_mask] = 0.0
                
            return filtered_batch, object_mask
        else:
            # Return only the selected objects, which changes the structure
            # This is more complex as we need to handle variable numbers of objects per event
            return self._extract_selected_objects(batch_data, object_mask), object_mask
    
    def _extract_selected_objects(self, batch_data: Dict, object_mask: torch.Tensor) -> Dict:
        """Extract only the selected objects from the batch, creating a new batch structure.
        
        This is more complex since different events might have different numbers of selected objects.
        
        Args:
            batch_data: Dictionary of batch tensors
            object_mask: Boolean mask of shape [batch_size, max_objects]
            
        Returns:
            Dictionary with restructured batch data containing only selected objects
        """
        batch_size = object_mask.shape[0]
        max_objects = object_mask.shape[1]
        
        # Count selected objects per event
        objects_per_event = object_mask.sum(dim=1)
        max_selected = objects_per_event.max().item()
        
        # Create storage for selected objects
        result = {}
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor) and v.shape[:2] == (batch_size, max_objects):
                # Create tensor with the right shape for selected objects
                shape = list(v.shape)
                shape[1] = max_selected
                result[k] = torch.zeros(shape, dtype=v.dtype, device=v.device)
                # Fill with padding token if it's the types tensor
                if k == 'types' or k == 'object_types':
                    result[k].fill_(self.padding_token)
            else:
                # Copy non-object tensors as is
                result[k] = v
        
        # Fill in selected objects for each event
        for event_idx in range(batch_size):
            event_mask = object_mask[event_idx]
            n_selected = objects_per_event[event_idx].item()
            
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor) and v.shape[:2] == (batch_size, max_objects):
                    selected_objects = v[event_idx, event_mask[event_idx]]
                    result[k][event_idx, :n_selected] = selected_objects
        
        return result

    def spatial_relationship_mask(self, 
                                 features: torch.Tensor, 
                                 types: torch.Tensor,
                                 relation_fn: Callable,
                                 exclude_padding: bool = True) -> torch.Tensor:
        """Create a mask based on spatial relationships between objects.
        
        Args:
            features: Object features tensor of shape [batch_size, max_objects, features]
            types: Object types tensor of shape [batch_size, max_objects]
            relation_fn: Function that defines the spatial relationship
                        Takes (obj1_features, obj2_features, obj1_types, obj2_types)
                        and returns a boolean tensor
            exclude_padding: Whether to exclude padding objects from consideration
            
        Returns:
            Boolean mask of shape [batch_size, max_objects]
        """
        batch_size, max_objects = types.shape
        mask = torch.zeros((batch_size, max_objects), dtype=torch.bool, device=types.device)
        
        # Apply padding mask if needed
        valid_objects = ~(types == self.padding_token) if exclude_padding else torch.ones_like(types, dtype=torch.bool)
        
        # For each event, check relationships between all pairs of objects
        for batch_idx in range(batch_size):
            for obj1_idx in range(max_objects):
                # Skip padding objects
                if not valid_objects[batch_idx, obj1_idx]:
                    continue
                    
                obj1_features = features[batch_idx, obj1_idx]
                obj1_type = types[batch_idx, obj1_idx]
                
                # Check relationship with all other objects
                for obj2_idx in range(max_objects):
                    if not valid_objects[batch_idx, obj2_idx] or obj1_idx == obj2_idx:
                        continue
                        
                    obj2_features = features[batch_idx, obj2_idx]
                    obj2_type = types[batch_idx, obj2_idx]
                    
                    # Apply the relation function
                    if relation_fn(obj1_features, obj2_features, obj1_type, obj2_type):
                        mask[batch_idx, obj1_idx] = True
                        break  # Once a relationship is found, we can move on
        
        return mask 