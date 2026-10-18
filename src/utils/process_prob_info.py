def standard_probinfo_weight(probinfo_dict, token_level="neg entropy", way="average"):
    """Return mean entropy from the stored negative entropy of a slice."""
    if token_level != "neg entropy" or way != "average":
        raise ValueError("SGA uses entropy with average slice aggregation")
    values = probinfo_dict["neg entropy"]
    return -sum(values) / len(values) if values else 0.0
