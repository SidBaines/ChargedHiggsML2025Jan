import torch
import numpy as np
from typing import List, Dict, Callable, Optional, Any
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class ResidualStreamExtractor:
    """
    Extracts residual stream activations at specified points in the model.
    """
    def __init__(self, model, points: List[str]):
        """
        Args:
            model: The transformer model.
            points: List of points to hook, e.g. ['block_0', 'block_1', 'final']
        """
        self.model = model
        self.points = points
        self.handles = []
        self.activations = {p: [] for p in points}

    def _make_hook(self, point):
        def hook(module, input, output):
            # output: [batch, n_obj, d_model]
            self.activations[point].append(output.detach().cpu())
        return hook

    def register_hooks(self):
        self.handles = []
        for i, block in enumerate(self.model.attention_blocks):
            if f'block_{i}' in self.points:
                h = block.register_forward_hook(self._make_hook(f'block_{i}'))
                self.handles.append(h)
        if 'final' in self.points:
            h = self.model.classifier.register_forward_hook(self._make_hook('final'))
            self.handles.append(h)

    def clear(self):
        self.activations = {p: [] for p in self.points}

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def extract(self, object_features, object_types):
        self.clear()
        self.register_hooks()
        with torch.no_grad():
            _ = self.model(object_features, object_types)
        self.remove_hooks()
        # Concatenate activations for each point
        return {p: torch.cat(self.activations[p], dim=0) for p in self.points}

def flatten_selected_objects(activations: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    Given activations [batch, n_obj, d_model] and mask [batch, n_obj], return [n_selected, d_model]
    """
    return activations[mask].cpu().numpy()

def fit_linear_probe(X: np.ndarray, y: np.ndarray, task: str = 'classification', max_iter=1000):
    """
    Fit a linear probe (logistic regression or ridge regression).
    """
    if task == 'classification':
        clf = LogisticRegression(max_iter=max_iter)
    else:
        clf = Ridge()
    clf.fit(X, y)
    return clf

def probe_accuracy(clf, X: np.ndarray, y: np.ndarray, task: str = 'classification'):
    if task == 'classification':
        return clf.score(X, y)
    else:
        y_pred = clf.predict(X)
        return np.corrcoef(y_pred, y)[0, 1]

def plot_pca(X: np.ndarray, y: np.ndarray, title: str = '', type_names: Optional[Dict[int, str]] = None):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.7)
    if type_names:
        legend_labels = [type_names.get(int(lbl), str(lbl)) for lbl in np.unique(y)]
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.title(title)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()
    plt.close()

def save_probe_results(results: Dict, path: str):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(results, f)

def load_probe_results(path: str) -> Dict:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f) 