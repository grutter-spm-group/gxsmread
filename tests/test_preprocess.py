import pytest
import xarray as xr
import preprocess as pp
import channel_config as cc

class TestPreProcess:
    filename = "./data/killburn_r17_LiNi_initial_scan042-M-Xp-Topo.nc"
    ds = xr.open_dataset(filename)
    # TODO: is this really one file!?!?!?different base names!
    mf_ds = xr.open_mfdataset(os.path.dirname(filename) + "*.nc")

    # TODO: Test multi-file for all of these too!

    @staticmethod
    def assert_dropped_old_dims(ds):
        with pytest.raises(KeyError):
            ds[pp.GXSM_DATA_VAR]
            ds[pp.GXSM_DATA_DIFFERENTIAL]

    def test_convert_floatfield_raw(self):
        channel_config_raw = cc.GxsmChannelConfig(name=None,
                                                  conversion_factor=1.0,
                                                  units='raw')
        new_ds = pp.convert_floatfield(self.ds, channel_config_raw)
        assert new_ds['Topo-Xp']
        assert new_ds['Topo-Xp'].attrs['units'] == 'raw'
        assert new_ds['Topo-Xp'].data == \
            (self.ds[pp.GXSM_DATA_VAR] *
             self.ds[pp.GXSM_DATA_DIFFERENTIAL]).data
        self.assert_drop_old_dims(new_ds)

    def test_convert_floatfield_physical(self):
        channel_config_physical = cc.GxsmChannelConfig(name='Topography',
                                                       conversion_factor=2.0,
                                                       units='Angstrom')
        new_ds = pp.convert_floatfield(self.ds, channel_config_physical)
        assert new_ds['Topo-Xp']
        assert new_ds['Topo-Xp'].attrs['units'] == 'Angstrom'
        assert new_ds['Topo-Xp'].data == \
            (self.ds[pp.GXSM_DATA_VAR] * self.ds[pp.DATA_DIFFERENTIAL]
             * 2.0).data
        self.assert_drop_old_dims(new_ds)

    def test_clean_up_metadata_default_data_vars(self):
        new_ds = pp.clean_up_metadata(ds)
        assert len(new_ds.data_vars) == len(pp.GXSM_KEPT_DATA_VARS)
        for var in pp.GXSM_KEPT_DATA_VARS:
            assert var in new_ds.data_vars and var not in new_ds.attrs
        assert len(new_ds.dims) == len(pp.GXSM_KEPT_DIMS)
        assert len(new_ds.attrs) == len(ds.attrs) + (len(ds.data_vars) -
                                                     len(pp.GXSM_KEPT_DATA_VARS))

    def test_clean_up_metadata_extra_data_var(self):
        added_var = 'rangex'  # Should exist in any gxsm file
        kept_vars = [pp.GXSM_KEPT_DATA_VARS, added_var]
        new_ds = pp.clean_up_metadata(ds, [added_var])
        assert len(new_ds.data_vars) == len(kept_vars)
        for var in kept_vars:
            assert var in new_ds.data_vars and var not in new_ds.attrs
        assert len(new_ds.dims) == len(pp.GXSM_KEPT_DIMS)
        assert len(new_ds.attrs) == len(ds.attrs) + (len(ds.data_vars) -
                                                     len(kept_vars))

    def test_is_gxsm_file(self):
        # TODO: Need a non-gxsm nc file..
