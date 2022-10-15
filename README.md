# Amortized Inference for Causal Structure Learning

**This `full` branch is intended for reproducing the results in
([Lorch et al., 2022](https://arxiv.org/abs/2205.12934)).
The code in this branch not maintained or updated anymore and may 
contain outdated notation and documentation.**

## Installation
To install the code, create a new conda environment with
```
conda env create --file environment.yml
```
This will automatically install the `avici` package (and all its requirements) 
and the local versions of `cdt` (bug-fixed) 
and `dibs` (enabling interventional data).

If you want to run the R baselines, refer to the end of this README for instructions.
For this, we provide an extended `environment_with_r.yml` config that also installs R.

The pretrained model checkpoints can be downloaded from 
this figshare project ([link](https://figshare.com/projects/Amortized_Inference_for_Causal_Structure_Learning/151008)).
## Results

All results in the paper are generated through the script `scripts/eval.py` via the following steps.
An experiment is configured by a folder in `experiments/` and consists of running four commands in sequence.
To test the pipeline, you can use the dummy experiment in `experiments/test/` and execute in sequence:
```
python scripts/eval.py test --train --submit
python scripts/eval.py test --data --submit
python scripts/eval.py test --methods --submit
python scripts/eval.py test --summary --submit
```
The results are saved in their own subfolder of the `results/` directory.
(The analogous experiment `experiments/test-baselines/` tests all baselines but requires the R installations.)
The `--submit` flag launches a given experiment, otherwise we get a printout of what is about the be run.

#### 1. Train the models
```bash
python scripts/eval.py linear-base --train --submit
python scripts/eval.py rff-base    --train --submit
python scripts/eval.py gene-base   --train --submit
```

For the ablation studies, we train additional models:
```bash
python scripts/eval.py ablation-rff-axis-d     --train --submit
python scripts/eval.py ablation-rff-axis-n     --train --submit
python scripts/eval.py ablation-rff-blocks-1   --train --submit
python scripts/eval.py ablation-rff-blocks-2   --train --submit
python scripts/eval.py ablation-rff-blocks-4   --train --submit
python scripts/eval.py ablation-rff-relational --train --submit
```
To use a trained model in a given experiment,
you have to update the path to the model checkpoint that AVICI (denoted as `ours`) uses
_in `methods.yaml` of the corresponding experiment config folder_.


If you want to use a pretrained model checkpoint (available [here](https://figshare.com/projects/Amortized_Inference_for_Causal_Structure_Learning/151008))
you need to specify the folder path of the unzipped download
in the `methods.yaml` file, as above.


#### 2. Generate all data sets
```bash
# generalization study
python scripts/eval.py linear-generalization/d=10  --data  --submit
python scripts/eval.py linear-generalization/d=20  --data  --submit
python scripts/eval.py linear-generalization/d=50  --data  --submit
python scripts/eval.py linear-generalization/d=100  --data  --submit
python scripts/eval.py linear-generalization/in-dist  --data  --submit
python scripts/eval.py linear-generalization/ood-graph  --data  --submit
python scripts/eval.py linear-generalization/ood-graph-function  --data  --submit
python scripts/eval.py linear-generalization/ood-graph-function-noise  --data  --submit

python scripts/eval.py rff-generalization/d=10  --data  --submit
python scripts/eval.py rff-generalization/d=20  --data  --submit
python scripts/eval.py rff-generalization/d=50  --data  --submit
python scripts/eval.py rff-generalization/d=100  --data  --submit
python scripts/eval.py rff-generalization/in-dist  --data  --submit
python scripts/eval.py rff-generalization/ood-graph  --data  --submit
python scripts/eval.py rff-generalization/ood-graph-function  --data  --submit
python scripts/eval.py rff-generalization/ood-graph-function-noise  --data  --submit

python scripts/eval.py gene-generalization/d=10  --data  --submit
python scripts/eval.py gene-generalization/d=20  --data  --submit
python scripts/eval.py gene-generalization/d=50  --data  --submit
python scripts/eval.py gene-generalization/d=100  --data  --submit
python scripts/eval.py gene-generalization/in-dist  --data  --submit
python scripts/eval.py gene-generalization/ood-graph  --data  --submit
python scripts/eval.py gene-generalization/ood-graph-function  --data  --submit
python scripts/eval.py gene-generalization/ood-graph-function-noise  --data  --submit

python scripts/eval.py transfer-to-linear  --data  --submit
python scripts/eval.py transfer-to-rff     --data  --submit

# benchmark
python scripts/eval.py linear-d=30  --data  --submit
python scripts/eval.py rff-d=30     --data  --submit
python scripts/eval.py gene-d=30    --data  --submit

python scripts/eval.py linear-d=100 --data  --submit
python scripts/eval.py rff-d=100    --data  --submit
python scripts/eval.py gene-d=100   --data  --submit

python scripts/eval.py linear-d=30-in-dist  --data --submit
python scripts/eval.py rff-d=30-in-dist     --data --submit
python scripts/eval.py gene-d=30-in-dist    --data --submit

# calibration and uncertainty quantification
python scripts/eval.py calibration-linear-d=30  --data  --submit
python scripts/eval.py calibration-rff-d=30     --data  --submit
python scripts/eval.py calibration-gene-d=30    --data  --submit

# sachs
python scripts/eval.py sachs    --data  --submit

# ablations
python scripts/eval.py ablation-EXP         --data --submit
python scripts/eval.py ablation-EXP-in-dist --data --submit

```

To run the `sachs` data, you need to unzip `real_data/sachs.zip` and place it in `real_data/`.
The `eval.py --data` call will simply copy this dataset into the experiment directory.


#### 3. Run all algorithms on all data sets (Note: need to install R requirements before)
```bash
# generalization study
python scripts/eval.py linear-generalization/d=10  --methods  --submit
python scripts/eval.py linear-generalization/d=20  --methods  --submit
python scripts/eval.py linear-generalization/d=50  --methods  --submit
python scripts/eval.py linear-generalization/d=100  --methods  --submit
python scripts/eval.py linear-generalization/in-dist  --methods  --submit
python scripts/eval.py linear-generalization/ood-graph  --methods  --submit
python scripts/eval.py linear-generalization/ood-graph-function  --methods  --submit
python scripts/eval.py linear-generalization/ood-graph-function-noise  --methods  --submit

python scripts/eval.py rff-generalization/d=10  --methods  --submit
python scripts/eval.py rff-generalization/d=20  --methods  --submit
python scripts/eval.py rff-generalization/d=50  --methods  --submit
python scripts/eval.py rff-generalization/d=100  --methods  --submit
python scripts/eval.py rff-generalization/in-dist  --methods  --submit
python scripts/eval.py rff-generalization/ood-graph  --methods  --submit
python scripts/eval.py rff-generalization/ood-graph-function  --methods  --submit
python scripts/eval.py rff-generalization/ood-graph-function-noise  --methods  --submit

python scripts/eval.py gene-generalization/d=10  --methods  --submit
python scripts/eval.py gene-generalization/d=20  --methods  --submit
python scripts/eval.py gene-generalization/d=50  --methods  --submit
python scripts/eval.py gene-generalization/d=100  --methods  --submit
python scripts/eval.py gene-generalization/in-dist  --methods  --submit
python scripts/eval.py gene-generalization/ood-graph  --methods  --submit
python scripts/eval.py gene-generalization/ood-graph-function  --methods  --submit
python scripts/eval.py gene-generalization/ood-graph-function-noise  --methods  --submit

python scripts/eval.py transfer-to-linear  --methods  --submit
python scripts/eval.py transfer-to-rff     --methods  --submit

# benchmark
python scripts/eval.py linear-d=30  --methods  --submit
python scripts/eval.py rff-d=30     --methods  --submit
python scripts/eval.py gene-d=30    --methods  --submit

python scripts/eval.py linear-d=100 --methods  --submit
python scripts/eval.py rff-d=100    --methods  --submit
python scripts/eval.py gene-d=100   --methods  --submit

python scripts/eval.py linear-d=30-in-dist  --methods --submit
python scripts/eval.py rff-d=30-in-dist     --methods --submit
python scripts/eval.py gene-d=30-in-dist    --methods --submit

# calibration and uncertainty quantification
python scripts/eval.py calibration-linear-d=30  --methods  --submit
python scripts/eval.py calibration-rff-d=30     --methods  --submit
python scripts/eval.py calibration-gene-d=30    --methods  --submit

# sachs
python scripts/eval.py sachs    --methods  --submit

# ablations
python scripts/eval.py ablation-EXP         --methods --submit
python scripts/eval.py ablation-EXP-in-dist --methods --submit

```

#### 4. Create summary and plots
```bash
# generalization study
python scripts/eval.py linear-generalization/d=10  --summary_sid  --submit
python scripts/eval.py linear-generalization/d=20  --summary_sid  --submit
python scripts/eval.py linear-generalization/d=50  --summary_sid  --submit
python scripts/eval.py linear-generalization/d=100  --summary_sid  --submit
python scripts/eval.py linear-generalization/in-dist  --summary_sid  --submit
python scripts/eval.py linear-generalization/ood-graph  --summary_sid  --submit
python scripts/eval.py linear-generalization/ood-graph-function  --summary_sid  --submit
python scripts/eval.py linear-generalization/ood-graph-function-noise  --summary_sid  --submit

python scripts/eval.py rff-generalization/d=10  --summary_sid  --submit
python scripts/eval.py rff-generalization/d=20  --summary_sid  --submit
python scripts/eval.py rff-generalization/d=50  --summary_sid  --submit
python scripts/eval.py rff-generalization/d=100  --summary_sid  --submit
python scripts/eval.py rff-generalization/in-dist  --summary_sid  --submit
python scripts/eval.py rff-generalization/ood-graph  --summary_sid  --submit
python scripts/eval.py rff-generalization/ood-graph-function  --summary_sid  --submit
python scripts/eval.py rff-generalization/ood-graph-function-noise  --summary_sid  --submit

python scripts/eval.py gene-generalization/d=10  --summary_sid  --submit
python scripts/eval.py gene-generalization/d=20  --summary_sid  --submit
python scripts/eval.py gene-generalization/d=50  --summary_sid  --submit
python scripts/eval.py gene-generalization/d=100  --summary_sid  --submit
python scripts/eval.py gene-generalization/in-dist  --summary_sid  --submit
python scripts/eval.py gene-generalization/ood-graph  --summary_sid  --submit
python scripts/eval.py gene-generalization/ood-graph-function  --summary_sid  --submit
python scripts/eval.py gene-generalization/ood-graph-function-noise  --summary_sid  --submit

python scripts/eval.py transfer-to-linear  --summary_sid  --submit
python scripts/eval.py transfer-to-rff     --summary_sid  --submit

# benchmark
python scripts/eval.py linear-d=30  --summary_sid  --submit
python scripts/eval.py rff-d=30     --summary_sid  --submit
python scripts/eval.py gene-d=30    --summary_sid  --submit

python scripts/eval.py linear-d=100 --summary_sid  --submit
python scripts/eval.py rff-d=100    --summary_sid  --submit
python scripts/eval.py gene-d=100   --summary_sid  --submit

python scripts/eval.py linear-d=30-in-dist  --summary_sid --submit
python scripts/eval.py rff-d=30-in-dist     --summary_sid --submit
python scripts/eval.py gene-d=30-in-dist    --summary_sid --submit

# calibration and uncertainty quantification
python scripts/eval.py calibration-linear-d=30  --summary_sid  --submit
python scripts/eval.py calibration-rff-d=30     --summary_sid  --submit
python scripts/eval.py calibration-gene-d=30    --summary_sid  --submit

# sachs
python scripts/eval.py sachs    --summary_sid  --submit

# ablations
python scripts/eval.py ablation-EXP         --summary_sid --submit
python scripts/eval.py ablation-EXP-in-dist --summary_sid --submit
```
You can use the `--summary` option instead of `--summary_sid` to only compute the SHD 
and avoid calling the R `SID` package, which requires R.


The illustration and generalization plots shown in the paper are created using the
experiment summaries saved in `results/` and the scripts 
`scripts/illustration_plot.py` 
and `scripts/generalization_plot.py`.

----
## Installation including R

If you want to run the all baselines, you need to install R and certain dependencies.
To install an R environment inside the conda environment, set up your environment with:
```
conda env create --file environment_with_r.yml
```

The R script we provide installs all R packages needed and can be run with:
```
Rscript rsetup.R
```
(The following R packages need to be installed:
`pcalg`, 
`kpcalg`, 
`bnlearn`, 
`sparsebn`, 
`D2C`, 
`SID`, 
`CAM`, 
`RCIT`.
We ship the archived `SID` and `CAM` packages in this repo, since it is not hosted by 
CRAN anymore. 
The `SID` source was downloaded from [here](https://cran.r-project.org/src/contrib/Archive/SID/).
The `CAM` source was downloaded from [here](https://cran.r-project.org/src/contrib/Archive/CAM/).
)
