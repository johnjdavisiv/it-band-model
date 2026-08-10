# An anatomically informed iliotibial band model

Most musculoskeletal models do not have an anatomically informed IT band. But this one does! Full-body model based on Rajagopal/Lai/Uhlrich2023 that also features a "real" IT band with fibers connecting from the gluteus maximus and tensor fascia lata.  

**Citation**: Davis, J. (2026). A full-body musculoskeletal model with an anatomically informed iliotibial band. SportRxiv. [DOI HERE]

📄 Read the preprint here: [LINK]

💿 Data: [https://doi.org/10.6084/m9.figshare.33190689](https://doi.org/10.6084/m9.figshare.33190689)

Models and code to reproduce the full-body direct collocation simulations of running supporting the results of my IT band modeling paper. 

## Installation


Use `conda` to set up `environment.yml`: 

```bash
conda env create -f environment.yml
conda activate osim46
```

Pulls OpenSim 4.6 from the opensim conda channel. You may need to point your geometry directory to your own OpenSim installation if you want to visualize the model in the GUI.  

**Required data**: 

The repository ships the experimental inputs for the core pipeline (`data/`, ~4 MB, from Hamner & Delp 2013). Two optional datasets live in [the paper's figshare folder](https://doi.org/10.6084/m9.figshare.33190689): 

* **input-data archive**: unzip to `data/emg/` - contains raw EMG + multi-speed GRFs (for segmenting gait cycles); only needed for the EMG validation component of the study
* **output-data archive**: unzip to `results/` - contains published RRA + MocoInverse outputs. These let you run the full results analysis without re-solving, which can take several hours


## To reproduce results

```bash
python scripts/01_scale.py # Scale to exemplar subject
python scripts/02_ik.py # Inverse kinematics
python scripts/03_rra.py # RRA - takes ~10-15 min

# MocoInverse solves - LONG! 
python -u scripts/04_moco_inverse.py --track all --cycle all
# Can take a few hours per solve and there are 10 gait cycles
#    (uses -u because output gets written to opensim .log files instead)

# Processing results
python scripts/05_aggregate.py # Gathers raw results
python scripts/06_hicks_report.py # Compare quality vs Hicks et al thresholds
python scripts/07_fiber_operating_length.py # Check F-L / F-V
python scripts/08_peak_summary.py #Check force peaks
python make_figures.py # One simple check plot per publication figure -> plots/

```

## Checking your reproduction against the paper

`make_figures.py` draws one plot per publication figure - though note these are simple "check figures" not publication-quality exports. Axis limits and color schemes and such may differ. 

## Citations and shout-outs

* [Eng et al. 2015](https://pubmed.ncbi.nlm.nih.gov/26162548/) for the raw SIMM files and carefully collected cadaveric data
* [Rajagopal et al. 2016](https://pubmed.ncbi.nlm.nih.gov/27392337/) for the base full-body model
* [Lai et al. 2017](https://pubmed.ncbi.nlm.nih.gov/28900782/) for improved glute function and the explicit call-out for the need for a better IT band
* [Dembia et al. 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493) for OpenSim Moco
* [Uhlrich 2022](https://pubmed.ncbi.nlm.nih.gov/35798755/) for the improved passive properties and inspiration for using constrained optimization to tweak model parameters
* [Hutchinson et al. 2023](https://link.springer.com/article/10.1007/s40279-021-01634-3) for the excellent anatomical overview
* Nick Bianco, Ross Miller, Brian Umberger, and many others on the [OpenSim Moco user forum](https://simtk.org/plugins/phpBB/indexPhpbb.php?group_id=1815&pluginname=phpBB) for many useful tips and tricks