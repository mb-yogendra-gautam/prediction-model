"""
Response Formatter Utilities
Provides utilities for formatting API responses, including rounding numeric values
"""

from typing import Any, Dict, List, Union


def round_numeric_values(data: Any, decimal_places: int = 2) -> Any:
    """
    Recursively round all float values in a data structure to specified decimal places.
    
    This function handles:
    - Dictionaries (recursively processes all values)
    - Lists (recursively processes all elements)
    - Floats (rounds to specified decimal places)
    - Integers (preserves as-is)
    - Other types (preserves as-is: strings, bools, None, etc.)
    
    Args:
        data: The data structure to process (dict, list, float, int, str, etc.)
        decimal_places: Number of decimal places to round to (default: 2)
    
    Returns:
        The same data structure with all float values rounded
    
    Examples:
        >>> round_numeric_values({'revenue': 35000.456789, 'count': 150})
        {'revenue': 35000.46, 'count': 150}
        
        >>> round_numeric_values([0.856789, 0.912345])
        [0.86, 0.91]
        
        >>> round_numeric_values({'nested': {'value': 12.3456}})
        {'nested': {'value': 12.35}}
    """
    if isinstance(data, dict):
        # Recursively process all values in the dictionary
        return {key: round_numeric_values(value, decimal_places) for key, value in data.items()}
    
    elif isinstance(data, list):
        # Recursively process all elements in the list
        return [round_numeric_values(item, decimal_places) for item in data]
    
    elif isinstance(data, float):
        # Round float values to specified decimal places
        return round(data, decimal_places)
    
    elif isinstance(data, int):
        # Preserve integers as-is (don't convert to float)
        return data
    
    else:
        # Preserve all other types as-is (str, bool, None, etc.)
        return data

