import numpy as np

# for a given numpy data type, get the number of bytes
def get_bytes(dtype):
    """
    Get the number of bytes for a given numpy data type.
    
    Parameters
    ----------
    dtype : str
        A numpy data type.
    
    Returns
    -------
    int
        The number of bytes.
    """
    return np.dtype(dtype).itemsize


def find_value(arr, value, mode='nearest'):
    """
    Find the index and value in an array that is at or closest to a given value.

    Parameters
    ----------
    arr : array-like
        The array to search.
    value : float
        The value to find.

    Optional
    --------
    mode : str ('nearest', 'exact')
        The mode to use when searching for the value.
        'nearest' finds the closest value. Only works for numeric arrays.
            If a non-numeric array is given, 'exact' will be used.
        'exact' finds the exact value.
    
    Returns
    -------
    idx : int
        The index of the found value.
    found_value : float
        The value that was found.
    """

    is_numeric = np.issubdtype(arr.dtype, np.number)

    if is_numeric:
        if mode == 'exact':
            idx = np.searchsorted(arr, value, side='left')
        elif mode == 'nearest':
            idx = np.argmin(np.abs(arr - value))
    else:
        idx = np.where(arr == value)[0]

    found_value = arr[idx]
    return idx, found_value

def format_indices(values):
    """
    Format numbers as indices.
    
    Parameters
    ----------
    values : np.ndarray
        The values to format.
        
    Returns
    -------
    indices : np.ndarray
        The formatted indices.
    """

    indices = np.round(values).astype(np.int64)
    indices = np.clip(indices, 0, None)
    indices = np.array(indices).ravel()
    return indices


