"""Contains read methods for a gxsm file.

This file contains the main methods to read from gxsm files and
convert to a NETCDF4 / HDF5 file.

For simplicity, we simply override the two main dataset accessor files
from xarray: open_dataset() and open_mfdataset() (allowing usage of
all the nifty xarray features, like dask).

Read the description of each method for more info.

Basic usage:
    ds = open_dataset(...)
    multifile_ds = open_mfdataset(...)
"""

import xarray
from pathlib import Path
from functools import partial
import channel_config as cc
from preprocess import preprocess


def open_mfdataset(paths: str | list[str | Path],
                   channels_config_path: str | Path | None = None,
                   use_physical_units: bool = True,
                   allow_convert_from_metadata: bool = True,
                   simplify_metadata: bool = True, **kwargs
                   ) -> xarray.Dataset:
    """Open multiple files as a single dataset.

    Wrapper on top of xarray.open_mfdataset(), for handling gxsm netcdf
    files. It opens the different .nc files as a single xarray.Dataset
    instance, expecting them to each correspond to a channel of the same
    collection.

    Note that gxsm .nc files contain the raw DAC counter data for each
    channel, rather than physical units. Additionally, many metadata
    attributes are saved as variables rather than attributes (see preprocess.py
    for further explanation). The added options allow the user to choose to
    'fix' these differences.

    Args:
        paths: either a string glob in the form "path/to/my/files/*.nc" or an
            explicit list of files to open. Paths can be given as strings or
            as pathlib Paths. Note that the xarray option of nested
            list-of-lists is not supported here.
        channels_config_path: either a string or Path to a 'channels config'
            toml file, containing the channel information for raw-to-physical
            conversion. See examples/channels_config.toml for an example.
        use_physical_units: whether or not to record the data in physical
            units. If true, we require 'conversion_factor'  and 'units'
            to exist (see optional exception below). Else, 'units' will be
            'raw' and 'conversion_factor' '1.0'.
        allow_convert_from_metadata: for some channels, there are hardcoded
            gxsm metadata attributes that contain the V-to-units conversion.
            If this attribute is true, we will use the metadata conversion
            as a fallback (i.e. if the config does not contain it).
        simplify_metadata: whether or not to convert all metadata variables
            to attributes.

    Returns:
        An xarray.Dataset instance, with each file's data being stored as a data
        variable named $channel, where $channel is the channel substring
        in the filepath.
    """
    channels_config_dict = cc.load_channels_config_dict(channels_config_path)
    partial_func = partial(preprocess, use_physical_units=use_physical_units,
                           allow_convert_from_metadata=allow_convert_from_metadata,
                           channels_config_dict=channels_config_dict)
    return xarray.open_mfdataset(paths, preprocess=partial_func, **kwargs)


def open_dataset(filename_or_obj: str | Path,
                 channels_config_path: str | Path | None = None,
                 use_physical_units: bool = True,
                 allow_convert_from_metadata: bool = True,
                 simplify_metadata: bool = True, **kwargs
                 ) -> xarray.Dataset:
    """Open and decode a dataset from a file or file object.

    Wrapper on top of xarray.open_dataset(), for handling gxsm nc
    files. It opens an .nc file as a single xarray.Dataset
    instance, with the 'z-data' stored as a data variable with name
    $channel, where $channel is the channel name for the file.

    Note that gxsm .nc files contain the raw DAC counter data for each
    channel, rather than phyiscal units. Additionally, many metadata
    attributes are saved as variables rather than attributes (see preprocess.py
    for further explanation). The added options allow the user to choose to
    'fix' these differences.

    Args:
        filename_or_obj: path of the file to open, can be given as a string
            or a pathlib Path. Note that other xarray type options are not
            supported here.
        channels_config_path: either a string or Path to a 'channels config'
            toml file, containing the channel information for raw-to-physical
            conversion. See examples/channels_config.toml for an example.
        use_physical_units: whether or not to record the data in physical
            units. If true, we require 'conversion_factor'  and 'units'
            to exist (see optional exception below). Else, 'units' will be
            'raw' and 'conversion_factor' '1.0'.
        allow_convert_from_metadata: for some channels, there are hardcoded
            gxsm metadata attributes that contain the V-to-units conversion.
            If this attribute is true, we will use the metadata conversion
            as a fallback (i.e. if the config does not contain it).
        simplify_metadata: whether or not to convert all metadata variables
            to attributes.

    Returns:
        An xarray.Dataset instance, with the file's data being stored as a data
        variable named $channel, where $channel is the channel name for the file
        in the file structure.
    """
    channels_config_dict = cc.load_channels_config_dict(channels_config_path)
    ds = xarray.open_dataset(filename_or_obj, **kwargs)
    return preprocess(ds, use_physical_units=use_physical_units,
                      allow_convert_from_metadata=allow_convert_from_metadata,
                      channels_config_dict=channels_config_dict)
