import torch
from typing import Dict, Tuple, List, Callable

def higgs_candidate_filter(delta_r_threshold: float = 0.8,
                          min_mass: float = 100.0,
                          max_mass: float = 150.0) -> Callable:
    """Filter objects that could form Higgs candidates based on proximity and mass.
    
    Args:
        delta_r_threshold: Maximum angular distance between jets
        min_mass: Minimum invariant mass for Higgs candidate
        max_mass: Maximum invariant mass for Higgs candidate
        
    Returns:
        Condition function for spatial_relationship_mask
    """
    def relation_fn(obj1_features, obj2_features, obj1_type, obj2_type):
        # Check if both are jets (3 = large-R jet, 4 = small-R jet)
        is_jet1 = obj1_type in [3, 4]
        is_jet2 = obj2_type in [3, 4]
        
        if not (is_jet1 and is_jet2):
            return False
        
        # Calculate ΔR
        px1, py1, pz1, e1 = obj1_features[:4]
        px2, py2, pz2, e2 = obj2_features[:4]
        
        pt1 = torch.sqrt(px1**2 + py1**2)
        pt2 = torch.sqrt(px2**2 + py2**2)
        
        eta1 = torch.asinh(pz1 / pt1)
        eta2 = torch.asinh(pz2 / pt2)
        
        phi1 = torch.atan2(py1, px1)
        phi2 = torch.atan2(py2, px2)
        
        delta_eta = eta1 - eta2
        delta_phi = torch.abs(phi1 - phi2)
        delta_phi = torch.min(delta_phi, 2*torch.tensor(torch.pi) - delta_phi)
        
        delta_r = torch.sqrt(delta_eta**2 + delta_phi**2)
        
        # Check proximity
        if delta_r >= delta_r_threshold:
            return False
        
        # Calculate invariant mass
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        e_sum = e1 + e2
        
        mass_squared = e_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
        mass = torch.sqrt(torch.abs(mass_squared)) * torch.sign(mass_squared)
        
        # Check mass range
        return min_mass <= mass <= max_mass
    
    return relation_fn

def boosted_object_filter(pt_threshold: float = 100.0,
                         type_list: List[int] = None) -> Callable:
    """Filter objects with high transverse momentum.
    
    Args:
        pt_threshold: Minimum pT for selection
        type_list: List of object types to consider
        
    Returns:
        Object filter function
    """
    def filter_fn(batch_data: Dict) -> torch.Tensor:
        features = batch_data.get('features', batch_data.get('x'))
        types = batch_data.get('types', batch_data.get('object_types'))
        
        # Calculate pT
        px = features[..., 0]
        py = features[..., 1]
        pt = torch.sqrt(px**2 + py**2)
        
        # Create mask for high pT objects
        pt_mask = pt > pt_threshold
        
        # Apply type filter if specified
        if type_list is not None:
            type_mask = torch.zeros_like(pt_mask, dtype=torch.bool)
            for obj_type in type_list:
                type_mask = type_mask | (types == obj_type)
            pt_mask = pt_mask & type_mask
            
        return pt_mask
    
    return filter_fn

def misreconstructed_objects_filter(model, 
                                  truth_key: str,
                                  confidence_threshold: float = 0.7,
                                  wrong_class: int = None) -> Callable:
    """Filter objects that are incorrectly reconstructed by the model.
    
    Args:
        model: The neural network model
        truth_key: Key for truth labels in batch data
        confidence_threshold: Minimum confidence for prediction
        wrong_class: If provided, only select objects misclassified as this class
        
    Returns:
        Object filter function
    """
    def filter_fn(batch_data: Dict) -> torch.Tensor:
        with torch.no_grad():
            # Get inputs
            features = batch_data.get('features', batch_data.get('x'))
            types = batch_data.get('types', batch_data.get('object_types'))
            truth = batch_data.get(truth_key)
            
            # Run model
            outputs = model(features, types)
            
            # Get predictions
            if outputs.dim() == 3:  # [batch, object, class]
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                conf = torch.max(probs, dim=-1).values
            else:  # [batch, class]
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                conf = torch.max(probs, dim=-1).values
                # Expand to object dimension if needed
                if truth.dim() > preds.dim():
                    preds = preds.unsqueeze(1).expand_as(truth[:,:,0])
                    conf = conf.unsqueeze(1).expand_as(truth[:,:,0])
            
            # Get truth labels
            if truth.dim() > 2:  # One-hot encoded
                true_labels = torch.argmax(truth, dim=-1)
            else:
                true_labels = truth
            
            # Find misclassified objects with high confidence
            misclassified = (preds != true_labels) & (conf > confidence_threshold)
            
            # Filter by specific wrong class if requested
            if wrong_class is not None:
                misclassified = misclassified & (preds == wrong_class)
                
            return misclassified
    
    return filter_fn 