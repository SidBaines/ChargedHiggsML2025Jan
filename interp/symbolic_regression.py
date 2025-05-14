import numpy as np

def symbolic_regression_on_bottleneck(
    X_query,
    X_key,
    Y,
    variable_names=None,
    regression_config=None,
):
    """
    Run symbolic regression to fit each bottleneck neuron as a function of query/key features.

    Args:
        X_query: [N, D_query] array of query object features.
        X_key: [N, D_key] array of key object features (optional, can be None).
        Y: [N, num_heads, bottleneck_dim] array of bottleneck activations.
        variable_names: List of variable names for regression input.
        regression_config: Dict of config for regression (e.g. which package, parameters).

    Returns:
        List of dicts, one per (head, bottleneck_dim), with:
            - 'expression'
            - 'score'
            - 'head'
            - 'neuron'
    """
    # Combine features
    if X_key is not None:
        X = np.concatenate([X_query, X_key], axis=-1)
        if variable_names is None:
            variable_names = [f'q_{i}' for i in range(X_query.shape[1])] + [f'k_{i}' for i in range(X_key.shape[1])]
    else:
        X = X_query
        if variable_names is None:
            variable_names = [f'x_{i}' for i in range(X_query.shape[1])]

    # Flatten Y to [N, num_heads * bottleneck_dim]
    N, num_heads, bottleneck_dim = Y.shape
    Y_flat = Y.reshape(N, num_heads * bottleneck_dim)

    # Choose regression package
    regressor = None
    if regression_config is None or regression_config.get('package', 'pysr') == 'pysr':
        try:
            from pysr import PySRRegressor
        except ImportError:
            raise ImportError("Please install pysr: pip install pysr")
        regressor = PySRRegressor(
            niterations=regression_config.get('niterations', 40) if regression_config else 40,
            binary_operators=regression_config.get('binary_operators', ["+", "-", "*", "/"]) if regression_config else ["+", "-", "*", "/"],
            unary_operators=regression_config.get('unary_operators', ["square", "sqrt"]) if regression_config else ["square", "sqrt"],
            variable_names=variable_names,
            maxsize=regression_config.get('maxsize', 30) if regression_config else 30,
            populations=regression_config.get('populations', 100) if regression_config else 100,
            population_size=regression_config.get('population_size', 30) if regression_config else 30,
            progress=True,
        )
    else:
        raise NotImplementedError("Only PySR is currently supported.")

    results = []
    for idx in range(num_heads * bottleneck_dim):
        y = Y_flat[:, idx]
        regressor.fit(X, y)
        expr = regressor.get_best()
        score = regressor.score(X, y)
        head = idx // bottleneck_dim
        neuron = idx % bottleneck_dim
        results.append({
            'expression': str(expr),
            'score': score,
            'head': head,
            'neuron': neuron,
        })
    return results 