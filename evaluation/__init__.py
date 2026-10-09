"""Offline leakage-safe experiments; never imported by the local runtime."""

CLASS_NAMES = ("Normal", "Yawning", "Eye Close")
CLASS_MAPPING = {name: index for index, name in enumerate(CLASS_NAMES)}
FEATURE_NAMES = ("EAR-R", "EAR-L", "MAR", "phi", "theta")
SEED = 20260927
