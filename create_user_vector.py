"""
Create normalized user preference vectors and compare them with cosine distance.

- Normalizes each parameter to [0, 1] using simple, capped transforms
  consistent with prior distance capping (rooms/roommates cap at 2, months cap at 36).
- Supports optional weighting of vector components.
"""
from math import sqrt


def _clamp(value, min_value, max_value):
    """Clamp a numeric value to the inclusive range [min_value, max_value]."""
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def normalize_rooms(x, cap=10):
    """Normalize desired rooms to [0, 1] by capping at cap and dividing by cap."""
    x = _clamp(x, 0, cap)
    return x / cap

    
def normalize_roommates(x, cap=10):
    """Normalize desired roommates to [0, 1] by capping at cap and dividing by cap."""
    x = _clamp(x, 0, cap)
    return x / cap


def normalize_budget(x, cap=200000):
    """Normalize budget to [0, 1] by clamping to [0, cap] and dividing by cap."""
    if cap <= 0:
        return 0.0
    x = _clamp(x, 0, cap)
    return x / float(cap)


def normalize_months(x, cap=36):
    """Normalize months to [0, 1] by clamping to [1, cap] and dividing by cap."""
    x = _clamp(x, 1, cap)
    return x / float(cap)

available_parameters = {
    'rooms': normalize_rooms,
    'roommates': normalize_roommates,
    'budget': normalize_budget,
    'months': normalize_months
}

#? Do we need weights?
parameter_weights = {
    'rooms': 1,
    'roommates': 1,
    'budget': 0.35,
    'months': 0.15
}

def create_user_vector(user:dict, parameters:list, caps:dict=None):
    """Create an unweighted normalized vector in the order of `parameters`."""
    vector = []
    caps = caps or {}
    for param in parameters:
        if param in available_parameters and param in user:
            normalizer = available_parameters[param]
            if param in ('budget', 'months'):
                # Pass optional cap if provided
                cap_value = caps.get(param)
                if cap_value is not None:
                    vector.append(normalizer(user[param], cap_value))
                else:
                    vector.append(normalizer(user[param]))
            else:
                vector.append(normalizer(user[param]))
        else:
            vector.append(0.0)
    return vector

def create_user_vector_with_weights(user:dict, parameters:list, weights:dict, caps:dict=None):
    """Create a weighted normalized vector (elementwise normalized_value * weight)."""
    base_vector = create_user_vector(user, parameters, caps)
    weighted_vector = []
    for i, param in enumerate(parameters):
        w = weights.get(param, 1.0)
        weighted_vector.append(base_vector[i] * w)
    return weighted_vector


def cosine_distance(vec1:list, vec2:list) -> float:
    """Compute cosine distance = 1 - cosine similarity for two equal-length vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 1.0
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a*a for a in vec1))
    norm2 = sqrt(sum(b*b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    return 1.0 - (dot / (norm1 * norm2))


if __name__ == "__main__":
    # Example users represented as dicts
    user_a = {
        'rooms': 1,
        'roommates': 1,
        'budget': 10000,
        'months': 12
    }
    user_b = {
        'rooms': 2,
        'roommates': 2,
        'budget': 15000,
        'months': 6
    }

    parameters = ['rooms', 'roommates', 'budget', 'months']
    weights = parameter_weights

    # Optional caps override (e.g., budget upper bound)
    caps = {'budget': 200000, 'months': 36}

    vec_a = create_user_vector(user_a, parameters, caps)
    vec_b = create_user_vector(user_b, parameters, caps)

    wvec_a = create_user_vector_with_weights(user_a, parameters, weights, caps)
    wvec_b = create_user_vector_with_weights(user_b, parameters, weights, caps)

    dist_unweighted = cosine_distance(vec_a, vec_b)
    dist_weighted = cosine_distance(wvec_a, wvec_b)

    print(f"User A vector (normalized): {vec_a}")
    print(f"User B vector (normalized): {vec_b}")
    print(f"Cosine distance (unweighted): {dist_unweighted}")
    print(f"Cosine distance (weighted): {dist_weighted}")
    
    print(f"Similarity score (unweighted): {round((1 - dist_unweighted)*100, 2)}%")
    print(f"Similarity score (weighted): {round((1 - dist_weighted)*100, 2)}%")