import pytest
import numpy as np
import xarray as xr
import glob
import gxsmread.read as read
import gxsmread.spec as spec
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


@pytest.fixture
def units_dict():
    return {'Index': '', 'ADC0-I': 'nA', 'ADC7': 'V', 'Zmon': 'Å', 'ZS': 'Å',
            'Block-Start-Index': ''}


@pytest.fixture
def names():
    # Note: our 'units' is made from a dict, so repeats are not duplicated.
    return ['Index', 'ADC0-I', 'ADC7', 'ADC0-I', 'ADC7', 'Zmon', 'ZS',
            'Block-Start-Index']


@pytest.fixture
def first_row_data():
    return [0, -2.762535436602e+01, -2.738731040376e+00, -2.762535436602e+01,
            -2.738731040376e+00, 3.850349997714e+03, 3.850349997714e+03,
            0.000000000000e+00]


def test_open_spec(units_dict, names, first_row_data):
    spec_filename = './tests/data/test007-VP003-VP.vpdata'
    spec_df = read.open_spec(spec_filename)

    assert (spec_df.columns.to_numpy() == names).all()
    assert spec_df.attrs[spec.KEY_UNITS] == units_dict
    assert np.allclose(spec_df[0:1].to_numpy(), np.array(first_row_data))
