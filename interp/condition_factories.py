import torch
import math
from typing import Callable, List, Tuple

def angular_proximity_condition(delta_r_threshold: float, 
                                type1: int = None, 
                                type2: int = None) -> Callable:
    """Creates a condition function that checks if objects are within a certain angular distance.
    
    Args:
        delta_r_threshold: Maximum angular distance (ΔR = √(Δη² + Δφ²))
        type1: Optional type of first object to consider
        type2: Optional type of second object to compare with
        
    Returns:
        Relation function for spatial_relationship_mask
    """
    def relation_fn(obj1_features, obj2_features, obj1_type, obj2_type):
        # Check object types if specified
        if type1 is not None and obj1_type != type1:
            return False
        if type2 is not None and obj2_type != type2:
            return False
        
        # Calculate ΔR using px, py, pz, E representation
        px1, py1, pz1, e1 = obj1_features[:4]
        px2, py2, pz2, e2 = obj2_features[:4]
        
        # Convert to pt, eta, phi
        pt1 = torch.sqrt(px1**2 + py1**2)
        pt2 = torch.sqrt(px2**2 + py2**2)
        
        eta1 = torch.asinh(pz1 / pt1)
        eta2 = torch.asinh(pz2 / pt2)
        
        phi1 = torch.atan2(py1, px1)
        phi2 = torch.atan2(py2, px2)
        
        # Calculate ΔR
        delta_eta = eta1 - eta2
        delta_phi = torch.abs(phi1 - phi2)
        # Handle phi wrap-around
        delta_phi = torch.min(delta_phi, 2*math.pi - delta_phi)
        
        delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)
        
        return delta_r < delta_r_threshold
    
    return relation_fn

def energy_ratio_condition(min_ratio: float = 0.5, 
                           max_ratio: float = 2.0,
                           type1: int = None, 
                           type2: int = None) -> Callable:
    """Creates a condition function that checks if objects have a specific energy ratio.
    
    Args:
        min_ratio: Minimum ratio of E1/E2
        max_ratio: Maximum ratio of E1/E2
        type1: Optional type of first object to consider
        type2: Optional type of second object to compare with
        
    Returns:
        Relation function for spatial_relationship_mask
    """
    def relation_fn(obj1_features, obj2_features, obj1_type, obj2_type):
        # Check object types if specified
        if type1 is not None and obj1_type != type1:
            return False
        if type2 is not None and obj2_type != type2:
            return False
        
        # Get energies
        e1 = obj1_features[3]
        e2 = obj2_features[3]
        
        # Calculate ratio
        ratio = e1 / e2
        
        return min_ratio <= ratio <= max_ratio
    
    return relation_fn

def combined_condition(conditions: List[Callable]) -> Callable:
    """Combines multiple conditions with logical AND.
    
    Args:
        conditions: List of condition functions
        
    Returns:
        Combined condition function
    """
    def combined_fn(obj1_features, obj2_features, obj1_type, obj2_type):
        for condition in conditions:
            if not condition(obj1_features, obj2_features, obj1_type, obj2_type):
                return False
        return True
    
    return combined_fn

def event_reco_type_condition(output_tensor: torch.Tensor, 
                             target_class: int, 
                             threshold: float = 0.5,
                             min_confidence: float = 0.0) -> torch.Tensor:
    """Creates a condition that selects events based on reconstruction type.
    
    Args:
        output_tensor: Model output tensor [batch, object, class]
        target_class: Target class to select
        threshold: Threshold for class probability
        min_confidence: Minimum confidence required
        
    Returns:
        Boolean mask of shape [batch_size]
    """
    # For reconstruction models with shape [batch, object, class]
    if output_tensor.dim() == 3:
        # Get probability of target class for each object
        if output_tensor.shape[-1] > 1:  # Multi-class case
            probs = torch.softmax(output_tensor, dim=-1)[:, :, target_class]
        else:  # Binary case
            probs = torch.sigmoid(output_tensor.squeeze(-1))
            if target_class == 0:
                probs = 1 - probs
                
        # Check if any object exceeds threshold with sufficient confidence
        confident_objects = (probs > threshold) & (torch.abs(probs - 0.5) > min_confidence)
        return confident_objects.any(dim=1)
    
    # For classification models with shape [batch, class]
    else:
        if output_tensor.shape[-1] > 1:  # Multi-class case
            probs = torch.softmax(output_tensor, dim=-1)[:, target_class]
        else:  # Binary case
            probs = torch.sigmoid(output_tensor.squeeze(-1))
            if target_class == 0:
                probs = 1 - probs
                
        return (probs > threshold) & (torch.abs(probs - 0.5) > min_confidence) 