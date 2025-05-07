"""
Hypothesis-specific plotting modules.

This package contains specialized visualization functions for different hypothesis types.
The functions are registered with the registry to allow them to be used through the registry.
"""

# Import all hypothesis plotters to ensure they're registered
from . import l8_concentration
from . import closed_lost_reason

from .single_dim import hypo_bar_scored

__all__ = ["hypo_bar_scored"] 