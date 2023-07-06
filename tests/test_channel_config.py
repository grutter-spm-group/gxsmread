# test loading proper toml
#   Topography:
#       - not in dict: standard
#          -> Should give Topo with nm
#       - in dict:
#           - only name
#               Should give 'name' with 'raw'
#           - all!
#               Should give 'name' with 'units' and conversion done
#
# Test set name:
#   - name provided -> should change it
#   - name not provided -> should use default
#
# Test raw:
#   - units should be raw and conversion 1.0
#
# Test physical:
#   - with conversion factor and units -> hunky dory!
#   - NOT allow convert from metadata:
#     - with conversion factor but no units -> error
#     - with no conversion factor but units -> error
#   - allow convert form metadata:
#     - channel not in MAP -> error
#     - channel in MAP but not in metadata -> error
#     - channel in MAP and metadata -> convert according to that!
#

import pytest
import xarray as xr
import channel_config as cc
import filename as fn
from utils import extract_numpy_data


@staticmethod
def assert_config(config: cc.GxsmChannelConfig,
                  scan_direction: str,
                  exp_name: str,
                  exp_units: str,
                  exp_factor: float ):
    """Simple config assertion."""
    assert config.name == fn.get_unique_channel_name(exp_name, scan_direction)
    assert config.units == exp_units
    assert config.conversion_factor == exp_factor


class TestTopography:
    # Special case of topography being the channel of interest
    filename = "./data/killburn_r17_LiNi_initial_scan042-M-Xp-Topo.nc"
    scan_direction = 'Xp'
    ds = xr.open_dataset(filename)
    file_attribs = fn.parse_gsxm_filename(filename)

    def test_no_dict(self):
        config = cc.GxsmChannelConfig(None, self.ds, self.file_attribs,
                                      use_physical_units=False,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction,
                      cc.GXSM_CHANNEL_TOPO, cc.GXSM_TOPO_UNITS,
                      cc.GXSM_TOPO_CONVERSION_FACTOR)

    def test_not_in_dict(self):
        config_dict = {'ADC0': {'name': 'banana',
                                'conversion_factor': 1.5,
                                'units': 'monkey'}}
        config = cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                      use_physical_units=False,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction,
                      cc.GXSM_CHANNEL_TOPO, cc.GXSM_TOPO_UNITS,
                      cc.GXSM_TOPO_CONVERSION_FACTOR)

    def test_only_name_in_dict(self):
        config_dict = {cc.GXSM_CHANNEL_TOPO: {'name': 'banana'}}
        config = cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                      use_physical_units=False,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction, config_dict['name'],
                      cc.DEFAULT_UNITS, cc.DEFAULT_CONVERSION_FACTOR)

    def test_all_in_dict(self):
        config_dict = {cc.GXSM_CHANNEL_TOPO: {'name': 'banana',
                                              'conversion_factor': 1.5,
                                              'units': 'monkey'}}
        config = cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                      use_physical_units=True,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction, config_dict['name'],
                      config_dict['units'], config_dict['conversion_factor'])


class TestSetName:
    filename = "./data/killburn_r17_LiNi_initial_scan041-Xp-ADC1.nc"
    scan_direction = 'Xp'
    ds = xr.open_dataset(filename)
    file_attribs = fn.parse_gsxm_filename(filename)

    def test_change_name(self):
        config_dict = {'ADC1': {'name': 'banana'}}
        config = cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                      use_physical_units=False,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction, config_dict['name'],
                      cc.DEFAULT_UNITS, cc.DEFAULT_CONVERSION_FACTOR)

    def test_keep_name(self):
        config = cc.GxsmChannelConfig(None, self.ds, self.file_attribs,
                                      use_physical_units=False,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction, self.file_attribs.channel,
                      cc.DEFAULT_UNITS, cc.DEFAULT_CONVERSION_FACTOR)


class TestPhysicalAndRaw:
    filename = "./data/killburn_r17_LiNi_initial_scan041-Xp-ADC1.nc"
    scan_direction = 'Xp'
    ds = xr.open_dataset(filename)
    file_attribs = fn.parse_gsxm_filename(filename)

    def test_raw(self):
        config_dict = {self.file_attribs.channel: {'conversion_factor': 1.5,
                                                   'units': 'monkey'}}
        config = cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                      use_physical_units=False,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction, config_dict['name'],
                      cc.DEFAULT_UNITS, cc.DEFAULT_CONVERSION_FACTOR)

    def test_factor_and_units(self):
        config_dict = {self.file_attribs.channel: {'name': 'banana',
                                                   'conversion_factor': 1.5,
                                                   'units': 'monkey'}}
        config = cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                      use_physical_units=True,
                                      allow_convert_from_metadata=False)
        assert_config(config, self.scan_direction, config_dict['name'],
                      config_dict['units'], config_dict['conversion_factor'])

    def test_factor_only_error(self):
        config_dict = {self.file_attribs.channel: {'conversion_factor': 1.5}}

        with pytest.raises(KeyError):
            cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                 use_physical_units=True,
                                 allow_convert_from_metadata=False)

    def test_units_only_error(self):
        config_dict = {self.file_attribs.channel: {'units': 'monkey'}}

        with pytest.raises(KeyError):
            cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                 use_physical_units=True,
                                 allow_convert_from_metadata=False)

    def test_wrong_key_error(self):
        config_dict = {'wrong_channel_name': {'units': 'monkey'}}

        with pytest.raises(KeyError):
            cc.GxsmChannelConfig(config_dict, self.ds, self.file_attribs,
                                 use_physical_units=True,
                                 allow_convert_from_metadata=False)

    def test_metadata_does_not_exist_for_channel_error(self):
        with pytest.raises(KeyError):
            cc.GxsmChannelConfig(None, self.ds, self.file_attribs,
                                 use_physical_units=True,
                                 allow_convert_from_metadata=True)

    def test_no_dict_physical_error(self):
        with pytest.raises(TypeError):
            cc.GxsmChannelConfig(None, self.ds, self.file_attribs,
                                 use_physical_units=True,
                                 allow_convert_from_metadata=False)


class TestPhysicalMetadata:
    filename = "./data/multichannel_file/r19_AuNP_LN084-Xm-ADC0mITunnel.nc"
    scan_direction = 'Xp'
    ds = xr.open_dataset(filename)
    file_attribs = fn.parse_gsxm_filename(filename)

    def test_metadata_not_file_somehow(self):
        # We actively delete the metadata data var!
        MAP = cc.GXSM_CHANNEL_METADATA_DICT
        self.ds = self.ds.drop_vars(MAP[self.file_attribs.channel]['name'])
        with pytest.raises(KeyError):
            cc.GxsmChannelConfig(None, self.ds, self.file_attribs,
                                 use_physical_units=True,
                                 allow_convert_from_metadata=True)

    def test_metadata_exists(self):
        MAP = cc.GXSM_CHANNEL_METADATA_DICT
        config = cc.GxsmChannelConfig(None, self.ds, self.file_attribs,
                                      use_physical_units=True,
                                      allow_convert_from_metadata=True)
        assert config.units == MAP[self.file_attribs.channel]['units']
        assert config.conversion_factor == \
            extract_numpy_data(self.ds[MAP[self.file_attribs.channel]['name']])
