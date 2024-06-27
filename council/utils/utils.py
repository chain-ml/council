from typing import Dict


def truncate_dict_values_to_str(data: Dict, max_length: int = 20):
    """
    Truncates dictionary values that are longer than max_length and returns a string representation.
    The truncated value shows both the start and end of the original value. Handles nested dictionaries recursively.

    Parameters:
    data (dict): The dictionary with values to be truncated.
    max_length (int): The maximum length of each value before truncation. Default is 20.

    Returns:
    str: A string representation of the dictionary with truncated values.
    """

    def truncate_value(value):
        if isinstance(value, dict):
            return truncate_dict_values_to_str(value, max_length)
        elif isinstance(value, list):
            return [truncate_value(item) for item in value]
        elif isinstance(value, str) and len(value) > max_length:
            half_length = (max_length - 3) // 2
            return value[:half_length] + "..." + value[-half_length:]
        else:
            return value

    truncated_items = []

    for key, value in data.items():
        truncated_value = truncate_value(value)
        truncated_items.append(f"{key}: {truncated_value}")

    return "{ " + ", ".join(truncated_items) + " }"
