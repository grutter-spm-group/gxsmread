import pytest
import numpy as np
import gxsmread.spec as spec


# ----- Metadata Testing ----- #
@pytest.fixture
def metadata():
    return """# view via: xmgrace -graph 0 -pexec 'title "GXSM Vector Probe Data: /home/nlt-user/Pictures/2025-02-06/test000-VP005-VP.vpdata"' -block /home/nlt-user/Pictures/2025-02-06/test000-VP005-VP.vpdata -bxy 2:4 ...
# GXSM Vector Probe Data :: VPVersion=00.02 vdate=20070227
# Date                   :: date=Wed Feb 26 15:26:03 2025
#
# FileName               :: name=/home/nlt-user/Pictures/2025-02-06/test000-VP005-VP.vpdata
# GXSM-Main-Offset       :: X0=0 Ang  Y0=0 Ang, iX0=-999999 Pix iX0=-999999 Pix
# DSP SCANCOORD POSITION :: NO MASTERSCAN SCAN COORDINATES N/A
# GXSM-DSP-Control-FB    :: Bias=0.1 V, Current=0 nA
# GXSM-DSP-Control-STS   :: #IV=1
# GXSM-DSP-Control-LOCKIN:: AC_amp=[ 0.02 V, 0, 14, 0],  AC_frq=4687.5 Hz,  AC_phaseA=0 deg,  AC_phaseB=90 deg,  AC_avg_cycles=32
# GXSM-Main-Comment      :: comment="nlt-user@nlt-afm Session Date: Wed Feb 26 15:10:56 2025 "
# Probe Data Number      :: N=505
# Data Sources Mask      :: Source=8388625
# X-map Sources Mask     :: XSource=268435472
#C """


@pytest.fixture
def key():
    return 'GXSM-DSP-Control-FB'

@pytest.fixture
def val():
    return 'Bias=0.1 V, Current=0 nA'


@pytest.fixture
def probe_pos():
    return ['0', '0', 'Ang']


@pytest.fixture
def probe_keys():
    return [spec.KEY_PROBE_POS_X,
            spec.KEY_PROBE_POS_Y,
            spec.KEY_PROBE_POS_UNITS]


@pytest.fixture
def date():
    return 'Wed Feb 26 15:26:03 2025'


@pytest.fixture
def date_key():
    return spec.KEY_DATE


@pytest.fixture
def filename():
    return '/home/nlt-user/Pictures/2025-02-06/test000-VP005-VP.vpdata'


@pytest.fixture
def filename_key():
    return spec.KEY_FILENAME


def test_extract_raw_metadata(metadata, key, val):
    lines = metadata.splitlines()
    md = spec.extract_raw_metadata(lines)
    assert len(md) == 12
    assert key in md
    assert md[key] == val


def test_parse_useful_metadata(metadata, probe_pos,
                               probe_keys, date, date_key,
                               filename, filename_key):
    lines = metadata.splitlines()
    md = spec.extract_raw_metadata(lines)
    useful_md = spec.parse_useful_metadata(md)

    keys = probe_keys + [date_key, filename_key]
    vals = probe_pos + [date, filename]

    for key, val in zip(keys, vals):
        assert key in useful_md
        assert useful_md[key] == val


# ----- Data Testing ----- #
@pytest.fixture
def spec_data():
    return r"""#C
#C Data Table             :: data=
#C Index	"ADC0-I (nA)"	"ADC0-I (nA)"	"Zmon (Å)"	"ZS (Å)"	Block-Start-Index
0	-2.763450991873e+01	-2.763450991873e+01	3.849527449114e+03	3.850349997714e+03	0.000000000000e+00
1	-2.763145806783e+01	-2.763145806783e+01	3.848469886628e+03	3.849354742453e+03	0.000000000000e+00
2	-2.763450991873e+01	-2.763450991873e+01	3.847412324142e+03	3.848359487191e+03	0.000000000000e+00
3	-2.764061362053e+01	-2.764061362053e+01	3.846472268599e+03	3.847364231929e+03	0.000000000000e+00
4	-2.763450991873e+01	-2.763450991873e+01	3.845532213055e+03	3.846368976668e+03	0.000000000000e+00
5	-2.764671732234e+01	-2.764671732234e+01	3.844474650569e+03	3.845373721406e+03	0.000000000000e+00
6	-2.763145806783e+01	-2.763145806783e+01	3.843417088083e+03	3.844378466145e+03	0.000000000000e+00
7	-2.763756176963e+01	-2.763756176963e+01	3.842594539483e+03	3.843383210883e+03	0.000000000000e+00
8	-2.763145806783e+01	-2.763145806783e+01	3.841536976997e+03	3.842387955622e+03	0.000000000000e+00
#C
#C END."""


@pytest.fixture
def expected_names():
    return ['Index', 'ADC0-I', 'ADC0-I', 'Zmon', 'ZS', 'Block-Start-Index']


@pytest.fixture
def expected_units():
    return ['', 'nA', 'nA', 'Å', 'Å', '']


@pytest.fixture
def expected_data():
    data = [[0, -2.763450991873e+01, -2.763450991873e+01, 3.849527449114e+03,
             3.850349997714e+03, 0.000000000000e+00],
            [1, -2.763145806783e+01, -2.763145806783e+01, 3.848469886628e+03,
             3.849354742453e+03, 0.000000000000e+00],
            [2, -2.763450991873e+01, -2.763450991873e+01, 3.847412324142e+03,
             3.848359487191e+03, 0.000000000000e+00],
            [3, -2.764061362053e+01, -2.764061362053e+01, 3.846472268599e+03,
             3.847364231929e+03, 0.000000000000e+00],
            [4, -2.763450991873e+01, -2.763450991873e+01, 3.845532213055e+03,
             3.846368976668e+03, 0.000000000000e+00],
            [5, -2.764671732234e+01, -2.764671732234e+01, 3.844474650569e+03,
             3.845373721406e+03, 0.000000000000e+00],
            [6, -2.763145806783e+01, -2.763145806783e+01, 3.843417088083e+03,
             3.844378466145e+03, 0.000000000000e+00],
            [7, -2.763756176963e+01, -2.763756176963e+01, 3.842594539483e+03,
             3.843383210883e+03, 0.000000000000e+00],
            [8, -2.763145806783e+01, -2.763145806783e+01, 3.841536976997e+03,
             3.842387955622e+03, 0.000000000000e+00]]
    return np.array(data, np.float32)


def test_extract_data(spec_data, expected_names, expected_units, expected_data):
    lines = spec_data.splitlines()
    received_names, received_units, received_data = spec.extract_data(lines)

    assert received_names == expected_names
    assert received_units == expected_units
    assert (received_data == expected_data).all()
