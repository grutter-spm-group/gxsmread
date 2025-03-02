"""Logic around reading spectroscopy files."""

import numpy as np


VISUALIZATION_HEADER = '# view'
COMMENT = '#C'

MD_STRIP_CHARS = '# '
MD_KEY_VAL_DELINEATOR = '::'

DATA_START = '#C Data Table'
DATA_END = '#C END.'
DATA_SEP = '\t'
UNIT_STRIP_CHARS = r'(")'

# Extracting useful metadata
SUBKEY_SEP = '='
POS_STRIP_CHARS = ' ,'
IN_KEYS_FILENAME = ['FileName', 'name']
IN_KEYS_DATE = ['Date', 'date']
IN_KEYS_PROBE_POS = ['GXSM-Main-Offset', ['X0', 'Y0']]

KEY_PROBE_POS_X = 'probe_position_x'
KEY_PROBE_POS_Y = 'probe_position_y'
KEY_PROBE_POS_UNITS = 'probe_position_units'
KEY_FILENAME = 'filename'
KEY_DATE = 'date'
KEY_UNITS = 'units'


def validate_spec_file(lines: list[str]):
    """Run quick checks to make sure this is a spec file."""
    assert lines[0].startswith(VISUALIZATION_HEADER)


def extract_raw_metadata(lines: list[str]) -> dict[str, str]:
    """Extract metadata from spec file lines into dict.

    Given the read lines from a spectroscopy file, extract the raw metadata
    and return it as a dict of metadata key and vals (both str).

    This metadata is of format:
        # KEY :: SUB_KEY1=... SUB_KEY2=...

    This method will extract the keys as independent attributes, with the full
    val string (consisting of various sub keys) stored as a metadata element.

    Args:
        lines: list[str] of read lines from a spectroscpy file.

    Returns:
        str:str key:val dict containing METADATA_KEY:METADATA_STR.
    """
    comment_indices = [i for i, v in enumerate(lines) if
                       v.startswith(COMMENT)]

    # Extract the metadata (OWN METHOD)
    raw_metadata = {}
    md_lines = lines[1:comment_indices[0]]

    # Skip lines without metadata key:vals
    md_lines = [line for line in md_lines if MD_KEY_VAL_DELINEATOR in line]

    for line in md_lines:
        kv = line.split(MD_KEY_VAL_DELINEATOR)
        k = kv[0].strip(MD_STRIP_CHARS)
        v = kv[1].strip()
        raw_metadata[k] = v

    return raw_metadata


def parse_useful_metadata(raw_metadata: dict[str, str]) -> dict[str, str]:
    """Parse raw metadata and extract some useful metadata.

    Particularly, we want to explicit:
    - Probe Position (x,y) where the spec file was collected, stored with
        KEY_PROBE_POS_X, KEY_PROBE_POS_Y, KEY_PROBE_POS_UNITS
    - Filename of the spec file, stored with KEY_FILENAME.
    - Date of the collection, stored with KEY_DATE.

    Args:
        raw_metadata: raw metadata extracted from spec file.

    Returns:
        new dict containing the parsed 'useful' data, with keys matching
            those indicated above.
    """
    useful_metadata = {}
    useful_metadata[KEY_DATE] = _extract_date(raw_metadata)
    useful_metadata[KEY_FILENAME] = _extract_filename(raw_metadata)

    coords, units = _extract_position(raw_metadata)
    useful_metadata[KEY_PROBE_POS_X] = coords[0]
    useful_metadata[KEY_PROBE_POS_Y] = coords[1]
    useful_metadata[KEY_PROBE_POS_UNITS] = units[0]
    return useful_metadata


def _extract_date(raw_metadata: dict[str, str]) -> str:
    date_subdata = raw_metadata[IN_KEYS_DATE[0]].split(SUBKEY_SEP)
    date_idx = [i for i, v in enumerate(date_subdata) if
                IN_KEYS_DATE[1] in v]
    assert len(date_idx) == 1
    date_idx = date_idx[0]
    return date_subdata[date_idx + 1]


def _extract_position(raw_metadata: dict[str, str]) -> (list[str], list[str]):
    pos_subdata = raw_metadata[IN_KEYS_PROBE_POS[0]].split(SUBKEY_SEP)

    coords = []
    units = []
    for pos in range(2):
        idx = [i for i, v in enumerate(pos_subdata) if
               IN_KEYS_PROBE_POS[1][pos] in v]
        idx = idx[0]

        # Expect: '0 Ang  ' or '0 Ang, '
        subdata = pos_subdata[idx + 1].split()
        coords.append(subdata[0].strip(POS_STRIP_CHARS))
        units.append(subdata[1].strip(POS_STRIP_CHARS))

    assert units[0] == units[1]
    return coords, units


def _extract_filename(raw_metadata: dict[str, str]) -> str:
    filename_subdata = raw_metadata[IN_KEYS_FILENAME[0]].split(SUBKEY_SEP)
    filename_idx = [i for i, v in enumerate(filename_subdata) if
                    IN_KEYS_FILENAME[1] in v]
    assert len(filename_idx) == 1
    filename_idx = filename_idx[0]
    return filename_subdata[filename_idx + 1]


def extract_data(lines: list[str]) -> (list[str], list[str], np.ndarray):
    r"""Extract data from spec file lines into names, units, and data.

    Given the read lines from a spectroscopy file, extract the data (as a
    numpy array), the channel names (as a list of strings), and the units
    (as a list of strings).

    The data if stored in this format:
        #C Data Table             :: data=
        #C Index\t"ADC0-I (nA)"\t [...]
        0\t1.1111\t2.222\t [...]
        [...]
        #C
        #C END.

    We use the first and last line expectations to constraint the data.
    We extract the units and channel names from the 2nd line.
    We extract all data between the 2nd line and the end.

    Args:
        lines: list[str] of read lines from a spectroscpy file.

    Returns:
        (list[str], list[str], np.ndarray) of:
        - list[str] of channel names
        - list[str] of units (empty string for Index/Block-start-index)
        - np.ndarray of all data.
    """
    data_start_indices = [i for i, v in enumerate(lines) if
                          v.startswith(DATA_START)]
    data_end_indices = [i for i, v in enumerate(lines) if
                        v.startswith(DATA_END)]
    data_range = (data_start_indices[0], data_end_indices[0])

    # Get data header
    data_header = lines[data_range[0] + 1].split(DATA_SEP)
    names, units = _extract_names_units(data_header)

    # Get data
    data_lines = lines[data_range[0] + 2:data_range[1] - 1]
    data = _extract_data(data_lines, len(names))

    return names, units, data


def _extract_names_units(data_header: str) -> (list[str], list[str]):
    # Remove '#C ' from first item (index)
    data_header[0] = data_header[0][3::]

    channel_names = [substr.split()[0].strip(UNIT_STRIP_CHARS)
                     for substr in data_header]
    units = [substr.split()[1].strip(UNIT_STRIP_CHARS)
             if UNIT_STRIP_CHARS[0] in substr else ''
             for substr in data_header]
    assert units[0] == ''  # Index should have no units

    return channel_names, units


def _extract_data(data_lines: list[str], num_cols: int) -> np.ndarray:
    data = [line.split(DATA_SEP) for line in data_lines]
    for line in data:
        assert num_cols == len(line)
    data = np.array(data, np.float32)
    return data
