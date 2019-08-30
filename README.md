# SubtLeNet
SUBsTructure-LEarning NETworks

## Installation

Either have your `PYTHONPATH` contain `SubtLeNet/python` or do:
```bash
cd SubtLeNet/python
pip install . [--user]
```
to make the package available system-wide.

## Preparing Samples for Training

### Converting .root files to .pkl 
To prepare samples for training, they must be converted from .root files to .pkl or .npy files.
This can be done using convert2.py (convert.py is deprecated).

#### convert_settings.json
convert2.py reads settings from a .json file
<ul>
  <li>Change `base` to whatever dir you have your .root files stored in</li>
  <li>Inside `samples`, the `name` of each sample is 1) what you pass the `--name` arg of convert2.py when you run it and 2) the prefix of the filenames of the .pkl files generated</li>
  <li>The 'samples' list paired with each name is a list of .root files to read, with the .root ending excluded</li>
  <li>`features` is a list of features/variable names/column names to read from the .root file</li>
  <li>.json files don't support comments, so I've established a convention of writing `"!features": [...]` to denote lists of features that aren't currently being used</li>
  <li>The keys, `j_pt` and `j_msd` are there so that convert2.py will grab those columns and store them in their own DataFrame, whether they're listed as a feature or not</li>
  <li>`cut_vars` are features/variables that are needed to make selections on the data, but aren't necessarily desired in the final DataFrame</li>
  <li>`signal_cut` and `background_cut` are selections to apply to the data. **Signal cut is always used unless `--background` is specified when you run convert2.py**</li>
  <li>`default` is a feature which, in the event no features are provided, convert2.py will grab. It will also grab any other features/columns with the same entry shape. So `"default": "fj_cpf_pt",` makes it so that if an empty feature list is given, convert2.py will grab all the cpf and ipf features, since they all have shape 1 x n_particles</li>
  <li>`per_part` **must** be specified if you want to read features with more than one number per event. Jet level features don't need this to be enabled, but cpf/ipf and secondary vertex features do.</li>
  <li>`n_particles`. The script will only take data from the first `n_particles` for each event</li>
</ul>

#### convert2.py
`python ./convert2.py -h`
```
usage: convert2.py [-h] [--out OUT] [--name NAME] [--json JSON] [--background]
                   [--verbosity [VERBOSITY]] [--dry]

optional arguments:
  -h, --help            show this help message and exit
  --out OUT             dir where output files will be stored
  --name NAME           name of sample to process
  --json JSON           json file controlling which
                        files/samples/features/cuts are used
  --background          use background cut instead of signal cut, also sets y
                        to 0
  --verbosity [VERBOSITY]
                        0-no printing, 1-print df.head() of output files
                        (default), 2-print info about x at different stages
                        and 1, 3-print the list of variables available in the
                        input .root file
  --dry                 if enabled, runs the whole program but doesn't save to
                        .pkl files
```

### Adding New Variables
add_vars.py can be used to add features/columns to the .pkl files that need to be calculated from existing features.
It takes one argument, `--json`, which must be used to point to a .json file with a filepath from which it can read the .pkl files.
plot_settings.json is good for this purpose, as this script and /python/subtlenet/plot.py usually are run around the same point in the overall process
Scroll to the bottom of this script to comment/uncomment function calls before use to make sure you add the variables you want.

### Making Plots 
/python/subtlenet/plot.py (not to be confused with /train/dazsle-tagger/plot.py, I know, I'm sorry) can be used to make histograms from .pkl files.
`python ./plot.py -h`
```
usage: plot.py [-h] [--out OUT] [--json JSON] [--trim [TRIM]] [--dpi [DPI]]

optional arguments:
  -h, --help     show this help message and exit
  --out OUT      filename of output pdf (the .pdf extension is optional)
  --json JSON    json file controlling the input files used
  --trim [TRIM]  slice this proportion of items from both ends of the Data
  --dpi [DPI]    dpi for output pdf
```
Scroll to the bottom of the script to comment/uncomment function calls.
In general, `make_hists(vars)` will make a histogram comparing each var in vars across the files specified in the .json file

## Training
