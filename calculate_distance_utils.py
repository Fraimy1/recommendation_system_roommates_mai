from math import exp

def norm_rooms(a, b):
    """
    Compute a normalized distance between two desired room counts.

    - Caps the penalty at a difference of 2 rooms to avoid over-penalizing outliers.
    - Maps absolute difference to [0, 1] by dividing by 2.

    Args:
        a: Desired number of rooms for user A.
        b: Desired number of rooms for user B.

    Returns:
        A float in [0, 1], where 0 means identical room counts and 1 means a difference of 2+ rooms.
    """
    return min(abs(a-b), 2) / 2.0

def norm_roommates(a, b):
    """
    Compute a normalized distance between desired roommate counts (others besides me).

    - Caps the penalty at a difference of 2 to keep the scale stable.
    - Maps absolute difference to [0, 1] by dividing by 2.

    Args:
        a: Desired number of roommates for user A.
        b: Desired number of roommates for user B.

    Returns:
        A float in [0, 1], where 0 means identical and 1 means a difference of 2+ roommates.
    """
    return min(abs(a-b), 2) / 2.0

def norm_budget(a, b):
    """
    Compute a budget distance using a ratio, robust to overall price scale.

    - Uses |a-b| / max(a, b) so 40k vs 50k yields 0.2, independent of currency scale.
    - If both are zero (degenerate), returns 1.0 (worst distance).

    Args:
        a: Monthly budget of user A (RUB).
        b: Monthly budget of user B (RUB).

    Returns:
        A float in [0, 1], where 0 means identical budget and values near 1 indicate large discrepancy.
    """
    # ratio distance, robust to scale
    if max(a, b) == 0:
        return 1.0
    return abs(a-b) / max(a, b)

def norm_months(a, b, cap=36):
    """
    Compute a normalized distance between desired stay durations in months.

    - Clamps both inputs to [1, cap] to avoid extreme influence.
    - Normalizes absolute difference by `cap` so result lies in [0, 1].

    Args:
        a: Desired months for user A.
        b: Desired months for user B.
        cap: Maximum months considered for normalization (default 36).

    Returns:
        A float in [0, 1] representing how far the durations are apart after clamping.
    """
    a = max(1, min(a, cap))
    b = max(1, min(b, cap))
    return abs(a-b) / cap

def numeric_distance(u, v, w_rooms=0.2, w_mates=0.3, w_budget=0.35, w_months=0.15):
    """
    Aggregate a weighted numeric distance between two users' preferences.

    Combines room count, roommate count, budget, and months-of-stay into a single
    distance via a weighted sum of normalized component distances.

    Args:
        u: Mapping with keys 'rooms', 'roommates', 'budget', 'months' for user A.
        v: Mapping with keys 'rooms', 'roommates', 'budget', 'months' for user B.
        w_rooms: Weight for room count distance.
        w_mates: Weight for roommates distance.
        w_budget: Weight for budget distance.
        w_months: Weight for months distance.

    Returns:
        A non-negative float where lower means more compatible by numeric prefs.
    """
    d = 0.0
    d += w_rooms   * norm_rooms(u['rooms'], v['rooms'])
    d += w_mates   * norm_roommates(u['roommates'], v['roommates'])
    d += w_budget  * norm_budget(u['budget'], v['budget'])
    d += w_months  * norm_months(u['months'], v['months'])
    return d

def combo_score(D, alpha=4.0):
    """
    Convert a distance into a bounded similarity score where higher is better.

    Uses an exponential decay: score = exp(-alpha * D). This keeps scores in (0, 1],
    is smooth and monotonic, and emphasizes differences near zero.

    Args:
        D: Any non-negative distance (e.g., from `numeric_distance`).
        alpha: Steepness parameter; larger values make the decay sharper.

    Returns:
        A float in (0, 1], with 1.0 for D=0 and approaching 0 as D grows.
    """
    return exp(-alpha * D)  # higher is better
