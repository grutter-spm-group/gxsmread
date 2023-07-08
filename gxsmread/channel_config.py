"""Channel Config File Access / Explanation.

This file holds the channel-config file logic: a method to read the file,
and a class to configure a channel for conversion. All of the 'options'
around converting to physical units is within the class's constructor.

The config file is a TOML file, which is read easily enough. Look in the
examples directory for a sample.

The options logic (and how the config file should be written) can be read
from the class documentation.
"""

from dataclasses import dataclass
import toml
import xarray
from . import utils
from . import filename as fn

# Hard-coded conversion info that may exist in the metadata
# TODO: investigate if other params are stored (e.g. dHertz2V)
# TODO: Validate whether ADC0mITunnel is the only way ADC0 is saved!?!?!
GXSM_CHANNEL_METADATA_DICT = {'ADC0mITunnel':
                              {'name': 'sranger_mk2_hwi_XSM_Inst_nAmpere2V',
                               'units': 'nA'}}

# Special case! We treat the topography in a unique way.
GXSM_CHANNEL_TOPO = 'Topo'
GXSM_TOPO_UNITS = 'nm'  # gxsm data stored in angstrom, converted to nm here
GXSM_TOPO_CONVERSION_FACTOR = 0.1

DEFAULT_UNITS = 'raw'
DEFAULT_CONVERSION_FACTOR = 1.0


@dataclass
class GxsmChannelConfig:
    """Class holds the attribs to convert to 'physical' units.

    You could argue this is overkill: toml.load() will already give us a
    multi-level dict! We mainly add this for 2 reasons:
        1. Encapsulate the config logic from elsewhere. We now have the
            'config' logic handled within here.
        2. Document what is expected from each channel in the .toml.

    Below, we explain how the toml file should be written (and the associated
    attributes). We use toml_dict to refer to the dict returned from reading
    the toml file.

    So: each 'channel' should look like this in the .toml:
    [$gxsm_channel_name$-$scan_direction$]
    name: ...
    conversion_factor: ...
    units: ...

    where, $gxsm_channel_name$ is the name of the channel as saved by gxsm
    (see GxsmFileAttribs descriptoin for more info).
    E.g. "Topo" for topography, "ADC0" for ADC0.

    Notes:
        - When we create the name for the *stored* variable, we must
            distinguish between scan directions (because we may have saved
            both directions on one channel). Thus, the saved data var will have
            a name that is:
                config.name + config.scan_direction.
        - Topography is a special case! It is somewhat 'handled' by gxsm and
            stored in Angstrom. We convert to nm here. However, we *do not*
            inhibit you from doing any conversion on your end. If you *do*
            override this behavior, make sure you consider that the base
            unit provided by gxsm is in Angstrom, and convert accordingly!

    Attributes:
        name: the name we should provide this channel in the final
            xarray.Dataset. If not in the toml_dict, we use $gxsm_channel_name$.
        conversion_factor: the factor to apply to the 'raw' units, to convert
            them to real units. Used in preprocess.py. If not in the toml_dict,
            we (optionally) first check if this data can be found in the
            metadata; if not, we raise ane exception.
        units: the true units we are saving in (if saving physical units). For
            the final created variable, we will add a 'units' attrib with this
            string. If not in the toml_dict, we (optoinally) first check if
            the data can be found in the metadata; if not, we raise an exception.
    """

    name: str
    conversion_factor: float
    units: str


def CreateGxsmChannelConfig(channels_config_dict: dict | None,
                            ds: xarray.Dataset,
                            file_attribs: fn.GxsmFileAttribs,
                            use_physical_units: bool,
                            allow_convert_from_metadata: bool
                            ) -> GxsmChannelConfig:
    """Create GxsmChannelConfig from input data.

    This contains the complicated 'config' setup logic. Read carefully!

    Args:
        channels_config_dict: a dict containing gxsm channel configuration
            data (some subset of these attributes) for each channel of
            interest.
        ds: the Dataset instance associated with this channel, used to
            extract metadata (if necessary)
        file_attribs: Gxsm file attributes, for easy access to 'channel'
            name.
        use_physical_units: whether or not to record the data in physical
            units. If true, we require 'conversion_factor'  and 'units'
            to exist (see optional exception below). Else, 'units' will be
            'raw' and 'conversion_factor' '1.0'.
        allow_convert_from_metadata: for some channels, there are hardcoded
            gxsm metadata attributes that contain the V-to-units conversion.
            If this attribute is true and toml_dict *does not contain* a
            conversion_factor, we will try to find a suitable conversion
            from the metadata.

    Returns:
        A GxsmChannelConfig instance.

    Raises:
        - KeyError if a key is missing from the channel_config_dict (or,
            if allowing metadata, the metadata attribs)
        - ValueError if the conversion factor provided / read is 0!
    """
    d = channels_config_dict
    c = file_attribs.channel
    MAP = GXSM_CHANNEL_METADATA_DICT

    # Recall the saved data var name must be unique (can have 2 scans
    # per channel)
    try:
        name = d[c]['name']
    except (KeyError, TypeError):
        name = c
    name = fn.get_unique_channel_name(name, file_attribs.scan_direction)

    # We allow the user to overide topography, but they must input
    # all parameters themselves
    if c == GXSM_CHANNEL_TOPO and (not d or GXSM_CHANNEL_TOPO not in d):
        conversion_factor = GXSM_TOPO_CONVERSION_FACTOR
        units = GXSM_TOPO_UNITS
    elif use_physical_units:
        # If no dict was passed, create an empty one (for code simplicity)
        d = {} if not d else d
        try:
            conversion_factor = d[c]['conversion_factor']
            units = d[c]['units']
        except (KeyError) as e:
            # Because of this, we *must* run this on the original ds
            # (we are grabbing metadata as data var, where we will be
            # turning it into an attr).
            if allow_convert_from_metadata and c in MAP and \
               MAP[c]['name'] in ds.data_vars:
                conversion_factor = utils.extract_numpy_data(
                    ds[MAP[c]['name']].values)
                units = MAP[c]['units']
                if conversion_factor == 0.0:
                    raise ValueError('Conversion factor for {} found in \
                    metadata, but was 0. Cannot continue.')
            else:
                raise type(e)(f'No conversion factor found for {c} (or '
                              'in metadata if permitted).') from e
    else:
        conversion_factor = DEFAULT_CONVERSION_FACTOR
        units = DEFAULT_UNITS
    return GxsmChannelConfig(name, conversion_factor, units)


def load_channels_config_dict(filename: str | None) -> dict:
    """Load config dict (currently a toml file)."""
    if filename:
        with open(filename, 'r') as f:
            return toml.load(f)
    else:
        return None
