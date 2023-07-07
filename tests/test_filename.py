import gxsmread.filename as fn


def test_main_file():
    str = "r19_AuNP_LN048-M-Xp-Topo.nc"
    file_attribs = fn.parse_gxsm_filename(str)
    assert file_attribs.file_base == "r19_AuNP_LN048"
    assert file_attribs.channel == "Topo"
    assert file_attribs.scan_direction == "Xp"
    assert file_attribs.is_main_file
    assert fn.get_unique_channel_name(file_attribs.channel,
                                      file_attribs.scan_direction) == "Topo-Xp"


def test_non_main_file():
    str = "r19_AuNP_LN048-Xm-ADC0mITunnel.nc"
    file_attribs = fn.parse_gxsm_filename(str)
    assert file_attribs.file_base == "r19_AuNP_LN048"
    assert file_attribs.channel == "ADC0mITunnel"
    assert file_attribs.scan_direction == "Xm"
    assert not file_attribs.is_main_file
    assert fn.get_unique_channel_name(file_attribs.channel,
                                      file_attribs.scan_direction) == \
           "ADC0mITunnel-Xm"
