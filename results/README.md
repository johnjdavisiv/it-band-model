# results/ -- pipeline outputs

This directory is empty by design. You can fill it in two ways:

1. **Download the output-data archive** from figshare deposit and unzip it here.

2. **Regenerate it** by running the pipeline (`scripts/01_scale.py` ... `scripts/04_moco_inverse.py`
   ... `scripts/08_peak_summary.py`). FYI the MocoInverse solves take a few hours per gait cycle and ther eare 10 total gait cycles! Note that MocoInverse may not be deterministic across platforms because of BLAS differences. 