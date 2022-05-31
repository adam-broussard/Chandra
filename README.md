# GalaxyTeam


## Installation

You can install from source by running:

```
git clone https://github.com/adam-broussard/GalaxyTeam
cd GalaxyTeam
python setup.py build
python setup.py install [--user]
```

## Usage
Once you have it installed, to do anything you'll need some data files which actually contain the X-ray scans. You can download them using:
```
>> from galaxyteam import dataset
>> dataset.download_dataset()
```
