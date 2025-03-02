# gxsmread

A small library to read gxsm data files.

## Gxsm Data Format Overview

### Scan Data Format

#### File Format

gxsm scan data files are stored in NetCDF3 format, with each channel stored as a separate .nc file in the format:

        $file_base$[-M-]-$scan_direction$-$channel$.nc
        
where:
- `$file_base$` is the user-provided filename;
- '-M-' indicates if it is the 'main file' of the recording (impling it
    contains the most metadata).
- `$scan_direction$` indicates the direction of scan while this data was
    recorded: Xp == --> (presumably forward), Xm == <-- (presumably
    backward).
- `$channel$` indicates the data channel. It will either be 'Topo' for
    topography, or '`ADC#[$extra_stuff$]`', indicating the explicit ADC
    channel it is collecting (these are the input ports of the
    collection module).
- brackets ([]) indicates an optional parameter.

#### Channel Data

For each file, the dimension of interest is stored in a data variable 'FloatField'; these are the RAW DAC counter data recorded. Additionally, a data field 'dz' is provided (defining the z differential); to convert to a 'pseudo-unit', we must multiply 'FloatField' by 'dz'. To convert to a physical unit, we must know the voltage-to-unit conversion corresponding to the ADC channel we have recorded.

gxsm does this because it allows *any* signal to be hooked into its input channels ADC0 - ADC7 (ADC = Analog to Digital Converter). It expects these signals to be a DC voltage, where the voltage level corresponds to a unit of interest. Thus, there is some conversion factor x, converting from DC V to a physical unit of interest.

Since gxsm knows nothing about the signal being fed, it cannot automatically convert it to the experimenter's units! gxsmread will perform these conversions automatically, provided the user includes a configuration file indicating the meaning of each channel of interest. This feature is enabled by default, but can be overriden by setting `$use_physical_units$` to False in the associated gxsm.read method.

#### Metadata

gxsm also stores a large amount of metadata on the experiment performed. To make this metadata even more clear, most of them are stored as data variables as well; this allows additional 'attributes' to be provided to clarify their meaning (e.g. `$long_name$`, `$Units$`).

However, this has the effect of creating *many* data variables for a given file, which may confuse the user. To simplify matters, gxsmread will by default convert all metadata variables to metadata attributes. This can be overriden by setting `$simplify_metadata$` to False in the associated gxsm.read method.

### Spectroscopy Data

gxsm spectroscopy files are stored in a plaintext .vpdata file. The file 
contains the following structure:

1.  Visualization Header: a single line indicates how to plot the file via a
terminal command.
2. Metadata: from the 2nd line up until the first '#C ' line, metadata 
attributes are indicadted in the format:
`$metadata_key$` :: `$subkey1$`=[...] `$subkey2$`=[...]

3. Channel Map Data: all of the potential save channels are indicated in a
tabular format, containing useful information about them (e.g. units).
4. Data: the saved data for the various *chosen* channels is stored in a
tabular format. The first line of this section contains the column names. All
remaining lines are tab-separated (`\t`) data rows. This section begins with 
`#C Data Table` and ends with `#C END.`.
5. Vector Probe Header list: an additional set of tabular data containing
additional information. I am not certain of the purpose of this.

Note that all spectroscopy files are saved in the same format, and there
appears to be no indicator for the 'type' of spectroscopy that was performed
(e.g. STS). The data channels taht are stored is user-specified, so this does
not give us any indication either.

## gxsmread Overview

### Scan Data Support

gxsmread tries to simplify gxsm data file reading, by allowing a mechanism to automatically read 1 or more gxsm .nc files into a single xarray dataset.

The expected inputs are:
- 1-n gxsm .nc files, each corresponding to a different recorded channel of a single experiment.
- [Optional] a .toml file indicating, for each channel:
  + the channel name (what it corresponds to).
  + the physical units that the channel's data respresents.
  + the conversion factor from recorded voltage to said physical units.

For a given channel:
- If a channel name is *not* provided, we keep the name gxsm saved it with (i.e. ADC#, except for the topography channel, which is saved as 'Topo').
- If physical units and the conversion factor are *not* provided, we leave the data as 'raw' and only perform the conversion to 'pseudo-units' (i.e. 'FloatField' * 'dz').

Note that Topography channels are a special case: we already have the information to convert them directly, and do so by default. However, the user can choose to override this behaviour by inputting a 'Topo' channel in their config .toml file.

### Spectroscopy Data support

gxsmread also supports reading spectroscopy files stored in the .vpdata format.
A given spectroscopy file is read, with the following saved into a pandas
DataFrame instance:
- The tabular data recorded, with channels/columns named.
- The units of each channel, stored in the DataFrame 'attrs' attribute, as a 
dict with key 'units'. This consists of key:val pairs containing 
CHANNEL_NAME:UNIT.
- All raw metadata, stored in the DataFrame 'attrs' attribute. We store each
`$metadata_key$:$string_containing_subkeys$` as a key:val pair in 'attrs'.
- A small number of desired metadata is further extracted from the subkeys.
See read:open_spec() for more info.

## Requirements

gxsmread depends on:
- xarray, which is the main accessor and stores the data as N-D labeled arrays.
- tomli, to read the toml files.
- netCDF4, to read the netCDF file format.
- Python 3.9+.
- (optionally) dask, for parallel loading.

## Installation

Note: this readme makes reference to poetry as the python virtual environment tool used. To use those commands, ensure you have installed poetry (see https://python-poetry.org/docs).

Clone this repository and install the dependencies in its pyproject.toml file.

If you are using poetry, you can do the following once in the cloned directory:

``` sh
poetry install
```

## Basic Usage

### Reading Scan Files

To open a single .nc file:

``` python
from gxsmread import read
[...]
path_to_file = '/path/to/file/with_filename.nc'
ds = read.open_dataset(path_to_file, channels_config_path,
                       use_physical_units=True,
                       simplify_metadata=True)
[...]

```

to open a multi-channel file:

``` python
from gxsmread import read
[...]
wildcard_path = 'path/to/files/*.nc'
ds = read.open_mfdataset(wildcard_path, channels_config_path,
                         use_physical_units=True,
                         simplify_metadata=True)
[...]
```

Note that open_mfdataset() support multi-threaded loading via dask by using the parameter parallel=True. gxsmread's method is a wrapper, and therefore does too.

### Reading Spectroscopy files

To open a single spectroscopy .vpdata file:

``` python
from gxsmread import read
[...]
path_to_file = '/path/to/file/with_filename.vpdata'
df = read.open_spec(path_to_file)
[...]

```
