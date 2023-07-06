"""gxsm filename parsing.

This file contains the logic to parse a gxsm filename into its component parts.
The included class allows easy access to these components (explained in the
class documentation).
"""

from dataclasses import dataclass
import os

GXSM_FILENAME_ATTRIB_SEPARATOR = '-'


@dataclass
class GxsmFileAttribs:
    """Class to hold the filename attributes, decoded.

    A gxsm filename is of the form:
        $file_base$[-M-]-$scan_direction$-$channel$.nc

    where:
        - $file_base$ is the user-provided filename;
        - '-M-' indicates if it is the 'main file' of the recording (impling it
            contains the most metadata);
        - $scan_direction$ indicates the direction of scan while this data was
            recorded: Xm == --> (presumably forward), Xp == <-- (presumably backward)
        - $channel$ indicates the data channel. It will either be 'Topo' for
            topography, or 'ADC#[$extra_stuff$]', indicating the explicit ADC
            channel it is collecting (these are the input ports of the collection
            module).
        - brackets ([]) indicates an optional parameter.
    """

    file_base: str
    channel: str
    scan_direction: str
    is_main_file: bool


def get_unique_channel_name(channel: str, scan_direction: str) -> str:
    """Provide a unique channel name.

    Since a file can contain data from the same channel twice
    (once per scan direction), we provide this helper method.
    """
    return channel + GXSM_FILENAME_ATTRIB_SEPARATOR + \
        scan_direction

def parse_gxsm_filename(filename: str) -> GxsmFileAttribs:
    """Helper to parse the gxsm filename into components.

    Args:
        filename: string to parse

    Returns:
        A GxsmFilename instance, indicating the filename attributes.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    substrs = basename.split(GXSM_FILENAME_ATTRIB_SEPARATOR)
    return GxsmFileAttribs(substrs[0], substrs[2], substrs[1],
                           len(substrs == 4))
