import xarray as xr
import glob
import gxsmread.read as read
import gxsmread.channel_config as cc
import gxsmread.preprocess as pp
import gxsmread.filename as fn
from . import test_preprocess as tpp


def flip_x_axis_floatfield(ds: xr.Dataset):
    """Flip X-axis for opposite scan direction data."""
    da = ds['FloatField'].copy(data=ds['FloatField'][:, :, :, ::-1])
    ds['FloatField'] = da
    return ds


def test_open_dataset():
    filename = "./tests/data/chigwell009-M-Xp-Topo.nc"
    std_ds = xr.open_dataset(filename)
    read_ds = read.open_dataset(filename, channels_config_path=None,
                                use_physical_units=True,
                                allow_convert_from_metadata=True,
                                simplify_metadata=True)

    config = cc.GxsmChannelConfig('Topo-Xp', cc.GXSM_TOPO_CONVERSION_FACTOR,
                                  cc.GXSM_TOPO_UNITS)
    kept_vars = ['Topo-Xp']
    old_vars_removed = pp.GXSM_KEPT_DATA_VARS

    tpp.assert_floatfield_conversion(std_ds, read_ds, config)
    tpp.assert_cleaned_metadata(std_ds, read_ds, kept_vars, old_vars_removed)


def test_open_mfdataset():
    # Perform a glob for all files...
    mf_filename = "./tests/data/chigwell009*.nc"

    mf_ds = read.open_mfdataset(mf_filename, channels_config_path=None,
                                use_physical_units=False,
                                allow_convert_from_metadata=False,
                                simplify_metadata=True)

    files = sorted(glob.glob(mf_filename))
    std_dses = [xr.open_dataset(file) for file in files]
    files_attribs = [fn.parse_gxsm_filename(file) for file in files]
    unique_channel_names = [fn.get_unique_channel_name(attrib.channel,
                                                       attrib.scan_direction)
                            for attrib in files_attribs]
    configs = []
    for uuid in unique_channel_names:
        if cc.GXSM_CHANNEL_TOPO in uuid:
            factor = cc.GXSM_TOPO_CONVERSION_FACTOR
            units = cc.GXSM_TOPO_UNITS
        else:
            factor = cc.DEFAULT_CONVERSION_FACTOR
            units = cc.DEFAULT_UNITS
        configs.append(cc.GxsmChannelConfig(uuid, factor, units))
    old_vars_removed = pp.GXSM_KEPT_DATA_VARS

    for (std_ds, config, kept_var, attribs) in zip(std_dses, configs,
                                                   unique_channel_names,
                                                   files_attribs):
        # xarray will properly handle flipped axes, but our crude
        # data equality check will not (unless we do so here)
        if attribs.scan_direction == fn.GXSM_BACKWARD_SCAN_DIR:
            std_ds = flip_x_axis_floatfield(std_ds)
        tpp.assert_floatfield_conversion(std_ds, mf_ds, config)
        if attribs.is_main_file:
            tpp.assert_cleaned_metadata(std_ds, mf_ds,
                                        unique_channel_names,
                                        old_vars_removed)
