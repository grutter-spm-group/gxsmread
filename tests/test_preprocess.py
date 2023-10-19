import pytest
import xarray as xr
import gxsmread.preprocess as pp
import gxsmread.channel_config as cc


def assert_floatfield_conversion(old_ds: xr.DataArray,
                                 new_ds: xr.DataArray,
                                 config: cc.GxsmChannelConfig):
    assert config.name in new_ds
    assert new_ds[config.name].attrs['units'] == config.units
    assert (new_ds[config.name].values ==
            (old_ds[pp.GXSM_DATA_VAR] *
             old_ds[pp.GXSM_DATA_DIFFERENTIAL] *
             config.conversion_factor).values).all()
    with pytest.raises(KeyError):
        new_ds[pp.GXSM_DATA_VAR]
    with pytest.raises(KeyError):
        new_ds[pp.GXSM_DATA_DIFFERENTIAL]


def assert_cleaned_metadata(old_ds: xr.DataArray,
                            new_ds: xr.DataArray,
                            kept_vars: list[str],
                            old_vars_removed: list[str] | None = None):
    # old_vars_converted are the old_ds variables that were removed before
    # metadata cleaning. This would happen if convert_floatfield() was run
    # before. That is why we provide this optional input.
    assert len(new_ds.data_vars) == len(kept_vars)
    for var in kept_vars:
        assert var in new_ds.data_vars and var not in new_ds.attrs
    assert len(new_ds.coords) == len(pp.GXSM_KEPT_COORDS)
    diff_vars_bw_old_and_new = old_vars_removed if old_vars_removed else \
        kept_vars
    len_kept_vars_coords = len(diff_vars_bw_old_and_new) + \
        len(pp.GXSM_KEPT_COORDS)
    assert len(new_ds.attrs) == (len(old_ds.attrs) +
                                 len(old_ds.data_vars) +
                                 len(old_ds.coords) -
                                 len_kept_vars_coords)


class TestPreProcess:
    # For reasons I don't understand, ds maintains state between
    # tests! This contradicts the pytest documentation:
    # https://docs.pytest.org/en/7.4.x/getting-started.html
    # (look for 'Group multiple tests in a class')
    # For now, just making a local copy at the start of each test
    filename = "./tests/data/chigwell009-M-Xp-Topo.nc"
    ds = xr.open_dataset(filename)


    def test_clean_floatfield(self):
        ds = self.ds
        new_ds = pp.clean_floatfield(ds.copy(deep=True))
        assert len(ds[pp.GXSM_DATA_VAR].dims) == 4
        assert len(new_ds[pp.GXSM_DATA_VAR].dims) == 2

    def test_clean_kept_coords(self):
        ds = self.ds
        new_ds = pp.clean_kept_coords(ds.copy(deep=True))
        for coord in pp.GXSM_KEPT_COORDS:
            assert ('units' in new_ds[coord].attrs and
                    new_ds[coord].attrs['units'] == cc.GXSM_TOPO_UNITS)
            assert (new_ds[coord].values ==
                    (ds[coord] * cc.GXSM_TOPO_CONVERSION_FACTOR).values).all()

    def test_convert_floatfield_raw(self):
        ds = pp.clean_floatfield(self.ds.copy(deep=True))
        ds = pp.clean_kept_coords(ds)

        cc_raw = cc.GxsmChannelConfig(name='Topo-Xp',
                                      conversion_factor=1.0,
                                      units='raw')
        new_ds = pp.convert_floatfield(ds, cc_raw)
        assert_floatfield_conversion(ds, new_ds, cc_raw)

    def test_convert_floatfield_physical(self):
        ds = pp.clean_floatfield(self.ds.copy(deep=True))
        ds = pp.clean_kept_coords(ds)

        cc_physical = cc.GxsmChannelConfig(name='Topo-Xp',
                                           conversion_factor=2.0,
                                           units='Angstrom')
        new_ds = pp.convert_floatfield(ds, cc_physical)
        assert_floatfield_conversion(ds, new_ds, cc_physical)

    def test_clean_up_metadata_default_data_vars(self):
        ds = pp.clean_floatfield(self.ds.copy(deep=True))
        ds = pp.clean_kept_coords(ds)

        kept_vars = pp.GXSM_KEPT_DATA_VARS
        new_ds = pp.clean_up_metadata(ds)
        assert_cleaned_metadata(ds, new_ds, kept_vars)

    def test_clean_up_metadata_extra_data_var(self):
        ds = pp.clean_floatfield(self.ds.copy(deep=True))
        ds = pp.clean_kept_coords(ds)

        added_vars = ['rangex']  # Should exist in any gxsm file
        kept_vars = pp.GXSM_KEPT_DATA_VARS + added_vars
        new_ds = pp.clean_up_metadata(ds, added_vars)

        assert_cleaned_metadata(ds, new_ds, kept_vars)

#    def test_is_gxsm_file(self):
#        # TODO: Need a non-gxsm nc file..
