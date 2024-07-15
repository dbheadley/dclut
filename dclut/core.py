import numpy as np
import os
import json
import copy
import xarray as xr
from tqdm import tqdm
from .utils import *

def create_dclut(bin_path, shape, dcl_path=None, dtype='int16', data_name='data', 
                 data_unit='au', scales=[]):
    """
    Create a dclut meta data file for a binary file. The meta file is formatted
    as a JSON file with the following fields.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file.
    shape : list, tuple, or numpy array of int
        Dimensionality of the LUT. All dimensions must be positive integers,
        except one which can be -1 to indicate an inferred length.
    
    Optional
    --------
    dcl_path : str
        Path to the dclut file. Default is bin_path with '_dclut.json' extension.
    dtype : string
        Data type of the binary file. Default is 'int16'.
    data_name : str
        Name of the data. Default is 'data'.
    data_unit : str
        Unit of the data. Default is 'au'.
    scales : list of dicts with scale settings. Each dict must have the following keys:
        name: str
            Names of the scales. Default is ['s0', s1', ... 'sN'], where N is the number
            of dimensions.
        dim : int
            Dimension to assign each scale. Default is [0, 1, ... N].
        unit : str
            Units of the scales. Default is ['au', 'au', ...].
        type : str
            Type of scale. These are:
            1. 'list' : A list of values, with one for each index.
            2. 'table' : An Nx2 array of values, where the first column is the index
                        and the second column is the value. Entries can be nonconsecutive.
                        Interpolation is required to fill in missing values.
            3. 'linear' : A slope factor and offset. The values are calculated as
                        value = index*slope + offset.
            4. 'index' : Just the indices are used, starting at 0 and going in steps of 1.
            Default is ['index', 'index', ...].
        val : numpy array
                Values for the scale. If corresponding scale_type is 'list', then the array
                must have the same length as the scale's corresponding dimension. If scale_type
                is 'linear', then values are [slope, offset]. If the scale_type is 'table', 
                then the array must be Nx2, where the first column is the index and the second 
                column is the value. The first row must be [0, value0] and the last row must be
                [length-1, valueN], where length is the length of the scale. If the scale_type 
                is 'index', then scale_vals is ignored (place None in the list).

    Returns
    -------
    dcl_path : str
        Path to the dclut file.
    """

    # Check if the binary file exists
    if not os.path.exists(bin_path):
        raise FileNotFoundError('Binary file {} not found'.format(bin_path))
    
    shape = np.array(shape).astype(np.int64)
    dim_num = shape.size

    # Check shape validity
    var_dim = np.where(shape == -1)[0]
    if len(var_dim) > 1:
        raise ValueError('Only one dimension can be inferred')
    
    # check file size and shape compatibility
    samp_bytes = get_bytes(dtype)
    file_bytes = os.path.getsize(bin_path)
    fixed_size = np.prod(shape[np.where(shape != -1)[0]])*samp_bytes
    var_size = file_bytes / fixed_size
    if not var_size.is_integer():
        raise ValueError('File size and shape are incompatible')
    
    shape[var_dim] = int(var_size)

    # convert shape to list for json serialization
    shape = shape.tolist()

    # Set default values
    if dcl_path is None:
        dcl_path = os.path.splitext(bin_path)[0] + '_dclut.json'

    # Initialize default index scales then add user scales
    for i in range(dim_num):
        scales.append({'name': 's{}'.format(i), 'dim': i, 'unit': 'au', 'type': 'index', 'val': None})

    # ensure all scale vals are numpy arrays
    for scale in scales:
        if scale['val'] is not None:
            scale['val'] = np.array(scale['val'])
    
    file_name = os.path.basename(bin_path)
    dcl = {}
    dcl['file'] = {'name': file_name, 'type': 'binary', 'shape': shape, 'params': {}}
    dcl['data'] = {'name': data_name, 'unit': data_unit, 'type': dtype, 'params': {}}
    dcl['scales'] = {}

    for sc in scales:
        curr_name = sc['name']
        curr_dim = sc['dim']
        curr_unit = sc['unit']
        curr_type = sc['type']
        curr_vals = sc['val']
        dcl['scales'][curr_name] = {'dim': curr_dim, 'unit': curr_unit, 'type': curr_type}

        if curr_type == 'list':
            if curr_vals.size != shape[curr_dim]:
                raise ValueError('{}\'s scale values do not match shape'.format(curr_name))
            dcl['scales'][curr_name]['values'] = curr_vals.tolist()

        elif curr_type == 'table':
            if (curr_vals[0,0] != 0) or (curr_vals[-1,0] != shape[curr_dim]-1):
                raise ValueError('{}\'s scale table is not properly formatted'.format(curr_name))
            dcl['scales'][curr_name]['values'] = curr_vals.tolist()

        elif curr_type == 'linear':
            if curr_vals.size != 2:
                raise ValueError('{}\'s linear scale is not properly formatted'.format(curr_name))
            dcl['scales'][curr_name]['values'] = curr_vals.tolist()

        elif curr_type == 'index':
            dcl['scales'][curr_name]['values'] = None

        else:
            raise ValueError('Unrecognized type for scale {}'.format(curr_name))
        
        dcl['scales'][curr_name]['params'] = {}

    with open(dcl_path, 'w') as f:
        json.dump(dcl, f)

    return dcl_path



        
class dclut():
    def __init__(self, path, verbose=False):
        """
        Initialize the dclut object.

        Parameters
        ----------
        path : str
            Path to a dclut file.
        
        Optional
        --------
        verbose : bool
            If True, show progress bars for loading and reading data. Default is False.
        """

        self.load(path=path)
        self._verbose = verbose


    def reset(self, dim=None):
        """
        Reset the selection for all data or a specific dim.

        Optional
        --------
        dim : int
            Dimension to reset. Default is all dimensions.
        """
        
        if dim is not None:
            if dim >= self.dim_num:
                raise ValueError('Dimension {} is out of range'.format(dim))
            self._selection[dim] = None

        else:
            self._selection = {d: None for d in range(self.dim_num)}
        
        return self
    
    
    def intervals(self, select, select_mode='union'):
        """
        Set the intervals for reading a across a scale.

        Parameters
        ----------
        select : dict
            Selections for each scale. The key is the scale name and the value is
            the intervals. Each row is an interval. The first column
            is the interval start (inclusive) and the second column is the interval
            end (exclusive). If multiple scales are provided, then all intervals must
            be the same size.

        Optional
        --------
        select_mode : str ('union', 'intersect', 'split')
            Mode for combining selections. 'union' takes the union of all selections within
            the selected dimension. 'intersect' takes the intersection of all selections within
            the selected dimension. 'split' breaks the selections into separate layers for each 
            selection. Default is 'union'.
        
        """

        scales, dim = self._validate_scales(select)
        intervals = self._validate_values(select)
        self._initialize_selection(dim)

        if intervals[0].size == 2:
            intervals = [i.reshape(1,2) for i in intervals]
        
        idxs = []
        num_intervals = intervals[0].shape[0]

        if num_intervals == 0:
            idxs = [np.array([])]
        else:
            for i in range(num_intervals):
                values = [interval[i] for interval in intervals]
                idxs.append(self._find_index(scales, values, mode='exact'))
    
        self._update_selection(dim, idxs, mode=select_mode)

        return self

    def points(self, select, select_mode='union', find_mode='nearest'):
        """
        Set specific points to read across a scale.

        Parameters
        ----------
        select : dict
            Selections for each scale. The key is the scale name and the value is
            the point or list of points to read. When multiple scales are provided,
            the number of points must be the same for each scale.
        
        Optional
        --------
        select_mode : str ('union', 'intersect', 'split')
            Mode for combining selections. 'union' takes the union of all selections within
            the selected dimension. 'intersect' takes the intersection of all selections within
            the selected dimension. 'split' breaks the selections into separate layers for each 
            selection. Default is 'union'.
        find_mode : str ('nearest', 'exact')
            Mode for point selection. 'nearest' selects the nearest value to the
            point. 'exact' selects the exact value. Default is 'nearest'.
        """

        
        scales, dim = self._validate_scales(select)
        points = self._validate_values(select)
        self._initialize_selection(dim)

        idxs = []
        num_points = points[0].size

        if num_points == 0:
            idxs = [np.array([])]
        else:
            for p in range(num_points):
                values = [point[p] for point in points]
                idxs.append(self._find_index(scales, values, mode=find_mode))
        
        self._update_selection(dim, idxs, mode=select_mode)

        return self

    def read(self, format='numpy'):
        """
        Read the data based on the current selections.
        
        Optional
        --------
        format : str ('numpy', 'xarray')
            Format of the data to return. 'numpy' returns a numpy array. 'xarray' 
            returns an xarray DataArray. The DataArray will have dimensions set to the
            first scales specified for each dimension in the dclut file. The remaining scales
            will be included as coordinates. Default is 'numpy

        Returns
        -------
        data : numpy array or xarray DataArray
            Data based on the current selections.
        """

        if self._test_empty_selection():
            raise ValueError('No selection made for any dimension')

        # fill any remaining unselected dimensions with all their values (i.e. select all)
        for dim in range(self.dim_num):
            if self._selection[dim] is None:
                self._selection[dim] = [np.arange(0, self.shape[dim])]
            
        data, idxs = self._select()

        if format == 'xarray':
            for i in range(len(data)):
                data[i] = xr.DataArray(data[i], dims=self._get_dims(), coords=self._get_coords(idxs[i]))

        return data
    
    def create_scale(self, scale):
        """
        Create a scale for the dclut object.

        Parameters
        ----------
        scale : dict
            Scale settings. The dict must have the following keys:
            name: str
                Names of the scale.
            dim : int
                Dimension of scale.
            unit : str
                Unit of the scale.
            type : str
                Type of scale. These are:
                1. 'list' : A list of values, with one for each index.
                2. 'table' : An Nx2 array of values, where the first column is the index
                            and the second column is the value. Entries can be nonconsecutive.
                            Interpolation is required to fill in missing values.
                3. 'linear' : A slope factor and offset. The values are calculated as
                            value = index*slope + offset.
                4. 'index' : Just the indices are used, starting at 0 and going in steps of 1.
            val : numpy array
                Values for the scale. If corresponding scale_type is 'list', then the array
                must have the same length as the scale's corresponding dimension. If scale_type
                is 'linear', then values are [slope, offset]. If the scale_type is 'table', 
                then the array must be Nx2, where the first column is the index and the second 
                column is the value. The first row must be [0, value0] and the last row must be
                [length-1, valueN], where length is the length of the scale. If the scale_type 
                is 'index', then scale_vals is ignored (place None in the list).
        
        Returns
        -------
        self : dclut object
            The dclut object with the new scale added.
        """

        sc_name = scale['name']
        sc_dim = scale['dim']
        sc_unit = scale['unit']
        sc_type = scale['type']
        sc_vals = np.array(scale['values'])

        if sc_name in self.dcl['scales']:
            raise ValueError('Scale {} already exists'.format(sc_name))
        
        self.dcl['scales'][sc_name] = {'dim': sc_dim, 'unit': sc_unit, 'type': sc_type}

        if sc_type == 'list':
            if sc_vals.size != self.shape[sc_dim]:
                raise ValueError('{}\'s scale values do not match shape'.format(sc_name))
            self.dcl['scales'][sc_name]['values'] = sc_vals

        elif sc_type == 'table':
            if (sc_vals[0,0] != 0) or (sc_vals[-1,0] != self.shape[sc_dim]-1):
                raise ValueError('{}\'s scale table is not properly formatted'.format(sc_name))
            self.dcl['scales'][sc_name]['values'] = sc_vals

        elif sc_type == 'linear':
            if sc_vals.size != 2:
                raise ValueError('{}\'s linear scale is not properly formatted'.format(sc_name))
            self.dcl['scales'][sc_name]['values'] = sc_vals

        elif sc_type == 'index':
            self.dcl['scales'][sc_name]['values'] = None

        else:
            raise ValueError('Unrecognized type for scale {}'.format(sc_name))
        
        self.dcl['scales'][sc_name]['params'] = {}

        return self

    def remove_scale(self, scale_name):
        """
        Remove a scale from the dclut object.
        
        Parameters
        ----------
        scale_name : str
            Name of the scale to remove.
            
        Returns
        -------
        self : dclut object
            The dclut object with the scale removed.
        """

        if scale_name not in self.dcl['scales']:
            raise ValueError('Scale {} not found'.format(scale_name))
        
        del self.dcl['scales'][scale_name]
    
        return self
    
    def save(self, path=None):
        """
        Save the dclut object to a json file.
        
        Parameters
        ----------
        path : str
            Path to save the dclut file. Default is the original path.

        Returns
        -------
        path : str
            Path to the saved dclut file.
        """

        # default to original path if none provided
        if path is None:
            path = self.dcl_path
        
        # convert all scales to lists for json serialization
        dcl = copy.deepcopy(self.dcl)
        for val in dcl['scales'].values():
            if val['type'] != 'index':
                val['values'] = val['values'].tolist()

        # save the dclut file
        with open(path, 'w') as f:
            json.dump(dcl, f)

        return path

    def load(self, path=None):
        """
        Load a dclut object from a json file.
        
        Parameters
        ----------
        path : str
            Path to the dclut file. Default is the original path, so
            the dclut object will be reloaded from the original file.
        
        Returns
        -------
        self : dclut object
            The dclut object loaded from the file.
        """

        if path is None:
            path = self.dcl_path

        self.dcl_path = path
        
        bin_dir = os.path.dirname(path)

        with open(path, 'r') as f:
            self.dcl = json.load(f)
        
        # convert all scales to numpy arrays after loading
        # from json
        for sn, sv in self.dcl['scales'].items():
            if sv['type'] == 'list':
                sv['values'] = np.array(sv['values'])
            elif sv['type'] == 'table':
                sv['values'] = np.array(sv['values'])
            elif sv['type'] == 'linear':
                sv['values'] = np.array(sv['values'])
            elif sv['type'] == 'index':
                sv['values'] = None
            else:
                raise ValueError('Unrecognized type for scale {}'.format(sn))
        
        self.bin_path = os.path.join(bin_dir, self.dcl['file']['name'])
        self.shape = tuple(self.dcl['file']['shape'])
        self.dim_num = len(self.shape)
        self._set_fio()
        self.reset()

        return self



    def _set_fio(self):
        self._fio = np.memmap(self.bin_path, dtype=self.dcl['data']['type'], mode='r', 
                              shape=self.shape)
        
    def _test_empty_selection(self):
        return all([sel is None for sel in self._selection.values()])
    
    def _get_dims(self):
        """
        Get the dimensions for generating xarray DataArray.
        
        Returns
        -------
        dims : list of str
            Dimensions for the xarray DataArray.
        """

        dims = []
        for dim in range(self.dim_num):
            # find the first scale specified for each dimension
            scale = [sn for sn, sv in self.dcl['scales'].items() if sv['dim'] == dim][0]
            dims.append(scale)
        
        return dims

    def _get_coords(self, indices):
        """
        Get the coordinates for generating xarray DataArray.
        
        Parameters
        ----------
        indices : list of numpy arrays
            Indices for the data for each dimension.

        Returns
        -------
        coords : dict
            Coordinates for the xarray DataArray.
        """

        # get dims to format dimensions and coordinates differently
        dims = self._get_dims()
        coords = {}
        for sn, sp in self.dcl['scales'].items():
            sv = self.scale_values(sn, indices[sp['dim']])
            if sn in dims:
                # dimension is just the scale values
                coords[sn] = sv 
            else:
                # coordinates are the corresponding dimension and scale values
                coords[sn] = (dims[sp['dim']], sv) 

        return coords

    def _update_selection(self, dim, idxs, mode='union'):
        """
        Update the selection for a dimension.

        Parameters
        ----------
        dim : int
            Dimension to update.
        idxs : list of numpy arrays
            Indices to update.
        
        Optional
        --------
        mode : str ('union', 'intersect', 'split')
            Mode for combining selections. 'union' takes the union of all selections within
            the selected dimension. 'intersect' takes the intersection of all selections within
            the selected dimension. 'split' breaks the selections into separate layers for each 
            selection. Default is 'union'.
        """
        
        if mode not in ['union', 'intersect', 'split']:
            raise ValueError('Unrecognized mode {}'.format(mode))
        
        if len(self._selection[dim]) == 0:
            if mode == 'split':
                self._selection[dim].extend(idxs)
            else:
                idxs = np.concatenate(idxs)
                # had to force conversion to int64 to avoid numpy defaulting to float64
                self._selection[dim].append(np.union1d(idxs,[]).astype('int64'))
        else:
            if mode == 'union': # combine new selection with existing selections
                idxs = np.concatenate(idxs)
                self._selection[dim] = [np.union1d(sel, idxs).astype('int64') for sel in self._selection[dim]]
            elif mode == 'intersect': # find the intersection of the new and existing selections
                idxs = np.concatenate(idxs)
                self._selection[dim] = [np.intersect1d(sel, idxs).astype('int64') for sel in self._selection[dim]]
            elif mode == 'split': # add new selections as separate layers
                self._selection[dim].extend(idxs)

    def _select(self):
        """
        Get the selections.

        Returns
        -------
        data : list of numpy arrays
            Data based on the selections.
        idxs : list of numpy arrays
            Indices of the data.
        """

        idxs = []
        data = []
        
        # test if all selection dims are the same size, or all are singleton except one
        sel_sizes = np.array([len(sel) for sel in self._selection.values()])
        if np.all(sel_sizes == 1):
            # if all selections are singleton, then just pass the single set of indices
            for dim in range(self.dim_num):
                idxs.append(self._selection[dim][0])
            idxs = [idxs]
        elif np.all(sel_sizes == sel_sizes[0]):
            # if all dimensions have the same number of selections, then return each set of selections
            for sel_i in range(sel_sizes[0]):
                idxs.append([self._selection[d][sel_i] for d in range(self.dim_num)])
        elif (np.unique(sel_sizes).size == 2) and (np.sum(sel_sizes == 1) == (self.dim_num-1)):
            # if only one dimension has multiple selections, then split the data along that dimension
            split_dim = np.where(sel_sizes != 1)[0][0]
            for sel_i in range(sel_sizes[split_dim]):
                idxs.append([])
                for d in range(self.dim_num):
                    if d == split_dim:
                        idxs[-1].append(self._selection[d][sel_i])
                    else:
                        idxs[-1].append(self._selection[d][0])
        else:
            raise ValueError('Selections are not compatible for split mode')    

        # get the data based on the indices
        if self._verbose:
            loop_iter = tqdm(range(len(idxs)))
        else:
            loop_iter = range(len(idxs))

        for i in loop_iter:
            idxs[i] = np.ix_(*idxs[i]) # convert to broadcastable indices
            data.append(self._fio[idxs[i]])
            self._set_fio() # reset the file pointer to free up memory

        return data, idxs


    def _find_index(self, scales, values, mode='nearest'):
        """
        Find the index for values spanning mulitple scales.
        
        Parameters
        ----------
        scales : list of str
            Name of the scales. Must be from the same dimension. Multiple scales
            can be provided when each is a list type. In that case, the indices 
            that minimizes the distance across all the scales will be returned.
        values : list of np.ndarray
            Values to find. List of values is used when multiple scales are provided.
            The number of values must be the same for each scale. Usually only one 
            value is provided, but if two are provided, the first is the start and
            the second is the end of an interval.

        Optional
        --------
        mode : str ('nearest', 'exact')
            Mode for finding the index when scale is a list type with numeric data. 
            'nearest' finds the index of the closest value. 'exact' finds the perfect 
            match for value. Default is 'nearest'.
        
        Returns
        -------
        value_idxs : numpy array of int
            Index of the values. If a value is present at multiple indices, then
            all are returned.
        """
        
        all_list = all([self.dcl['scales'][s]['type'] == 'list' for s in scales])

        if all_list: # if all scales are list type, find the index using all values
            # get values for each scale
            scale_values = []
            for scale in scales:
                scale_values.append(self.scale_values(scale))

            # replace nan values with infinity for comparison
            scale_values = [np.where(np.isnan(sv), np.inf, sv) for sv in scale_values]

            values_num = values[0].size
            scales_num = len(scales)
            value_idxs = []

            # if only one value, find the index/indices that are closest/match
            if values_num == 1: 
                dist = []
                for s in range(scales_num):
                    if scale_values[s].dtype == np.number:
                        dist.append(np.abs(scale_values[s] - values[s]))
                    else: # forces exact match when scales are non-numeric
                        dist.append(np.where(scale_values[s] == values[s], 0, np.inf))

                # collapse distances across scales and find minimum
                dist_all = np.sum(np.vstack(dist), axis=0)
                dist_min = np.min(dist_all)
                value_idxs = np.where(dist_all == dist_min)[0]

                # ensure a minimum was present and meets matching criteria
                if any(dist_all[value_idxs] == np.inf):
                    raise ValueError('Value not found in scale')
                elif (mode == 'exact') and any(dist_all[value_idxs] != 0):
                    raise ValueError('Value not found in scale with exact mode')
            elif values_num == 2:
                valid = []
                for s in range(scales_num):
                    if scale_values[s].dtype == np.number:
                        valid.append(scale_values[s] >= values[s][0])
                        valid.append(scale_values[s] < values[s][1])
                    else:
                        raise ValueError('Interval selection not supported for non-numeric scales')
                    valid_all = np.vstack(valid)
                    valid_all = np.all(valid_all, axis=0)
                    value_idxs = np.where(valid_all)[0]
            else:
                raise ValueError('Values must be one or two values')
            
            value_idxs = np.array(value_idxs).ravel()

        else: # if scales are not all list type, find the index for the single value
            scale_type = self.dcl['scales'][scales[0]]['type']
            if scale_type == 'table':
                table = self.dcl['scales'][scales[0]]['values']
                value_idxs = np.interp(values[0], table[:,1], table[:,0])
            elif scale_type == 'linear':
                slope, offset = self.dcl['scales'][scales[0]]['values']
                value_idxs = (values[0] - offset) / slope
            elif scale_type == 'index':
                value_idxs = values[0]
            else:
                raise ValueError('Unrecognized type for scale {}'.format(scales[0]))
        
            if value_idxs.size == 2:
                value_idxs = np.arange(value_idxs[0], value_idxs[1])

        value_idxs = format_indices(value_idxs)
        return value_idxs


    def _validate_scales(self, select):
        """
        Validate the scales for the dclut file.

        Parameters
        ----------
        select : dict
            Selections for each scale. The key is the scale name and the value is
            the point or list of points to read.

        Returns
        -------
        scales : list of str
            Validated scales.
        dim : int
            Dimension of the scales.
        """

        scales = list(select.keys())

        # check that scales are in the dclut file
        for scale in scales:
            if scale not in self.dcl['scales']:
                raise ValueError('Scale {} not found in dclut file'.format(scale))
        
        # Multiple scales can be used only if they are all the 'list' type
        if len(scales) > 1:
            if not all([self.dcl['scales'][s]['type'] == 'list' for s in scales]):
                raise ValueError('Multiple scales must be of list type')
        
        # Ensure all scales are from the same dimension
        scale_dims = [self.dcl['scales'][scale]['dim'] for scale in scales]
        if len(set(scale_dims)) > 1:
            raise ValueError('Scales must be from the same dimension')
        
        return scales, scale_dims[0]
    
    def _validate_values(self, select):
        """
        Validate the values for the scales.
        
        Parameters
        ----------
        select : dict
            Selections for each scale. The key is the scale name and the value is
            the point or list of points to read.

        Returns
        -------
        values : list of numpy arrays
            Validated values.
        """

        # ensure consistent formatting of values
        values = []
        for sn in select.keys():
            if select[sn] is None:
                values.append(np.array([]))
            else:
                values.append(np.array(select[sn]))

        # check that all values are the same shape
        shp = values[0].shape
        if not all([v.shape == shp for v in values]):
            raise ValueError('Values must have the same shape')
        
        return values
        
    
    def _initialize_selection(self, dim):
        """
        Format and initialize the selection for a dimension.
        """
        
        if self._selection[dim] is None:
            self._selection[dim] = []


    def scale_values(self, scale, indices=None):
        """
        Get the values for a scale.
        
        Parameters
        ----------
        scale : str
            Name of the scale.
        indices : array-like
            Indices to return. Default is all values.

        Returns
        -------
        values : numpy array
            Values for the scale.
        """

        scale_type = self.dcl['scales'][scale]['type']
        scale_values = self.dcl['scales'][scale]['values']
        scale_dim = self.dcl['scales'][scale]['dim']
        dim_len = self.dcl['file']['shape'][scale_dim]

        if indices is None:
            indices = np.arange(dim_len)
        else:
            indices = np.array(indices)
        
        if scale_type == 'list':
            values = np.array(scale_values)[indices]
        elif scale_type == 'table':
            values = np.interp(indices, np.array(scale_values)[:,0],
                               np.array(scale_values)[:,1])
        elif scale_type == 'linear':
            slope, offset = scale_values
            values = indices*slope + offset
        elif scale_type == 'index':
            values = indices
        else:
            raise ValueError('Unrecognized type for scale {}'.format(scale))
        
        return values.flatten()
    

    def windows(self, dim, window_size=1, window_starts=None):
        """
        Set the window size and starts for reading across a dimension.
        Ensures that every windowed selection is the same size. Determines
        the number of samples in the windowed selection based on the average
        number of samples returned from each windowed selection.

        
        Parameters
        ----------
        dim : str
            Dimension name.
        window_size : int
            Size of the window. Default is 1.
        window_starts : list
            list of window starts. Default is the first value"""

        ## TO DO ##

    #def read(self, **kwargs)
    # create a dunder print method that prints the dclut file parameters
    def __str__(self):
        """"
        Print the dclut file parameters
        
        Return
        ------
        s : str
            String representation of the dclut file parameters.
        """
        
        s = 'File: {}\n'.format(self.dcl['file']['name'])
        s += 'Type: {}\n'.format(self.dcl['file']['type'])
        s += 'Shape: {}\n'.format(self.dcl['file']['shape'])
        s += '----------------\n\n'
        s += 'Data: {} ({})\n'.format(self.dcl['data']['name'], self.dcl['data']['unit'])
        s += 'Type: {}\n'.format(self.dcl['data']['type'])
        s += 'Scales:\n'
        for sn, sv in self.dcl['scales'].items():
            s += '\n'
            s += '  {} ({})\n'.format(sn, sv['unit'])
            s += '  ' + '-'*len(sn) + '\n'
            s += '  dimension: {}\n'.format(sv['dim'])
            s += '  type: {}\n'.format(sv['type'])
            dim_len = self.dcl['file']['shape'][sv['dim']]
            if sv['type'] == 'index':
                scale_values = np.array([0, dim_len-1])
            elif sv['type'] == 'linear':
                scale_values = np.array([sv['values'][0]*dim_len, 0])+ sv['values'][1]
            elif sv['type'] == 'table':
                scale_values = np.array(sv['values'][:,1])
            else:
                scale_values = sv['values']
            if scale_values.dtype == np.number:
                s += '  min: {}\n'.format(np.nanmin(scale_values))
                s += '  max: {}\n'.format(np.nanmax(scale_values))
            else:
                s += '  values: {}\n'.format(scale_values)
        return s
            