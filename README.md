# Chest X-Ray Classification with Flexible Region Guidance

CS421: Selected Topics in Computer Science

King Faisal University

Group 5, Section 63

Track 1: Research Track

## Contributors

- Helh Bader Alhussain
- Alzahra Ali Alabdullah

Instructor: Dr. Hala Hamdoun

## Project Overview

This project investigates whether distance-weighted spatial guidance
can preserve classification performance and localization alignment
when training bounding-box annotations are imperfect.

The task is binary classification of chest radiographs with and
without annotated pneumonia-like opacities. It is not intended
to establish a clinical diagnosis.

Four training approaches are compared:

- Classification-only DenseNet-121
- Binary-membership guidance
- Fixed-padding guidance
- Distance-weighted guidance (proposed method)

The guided methods are evaluated under four training annotation
conditions: unchanged, translated, shrunk and enlarged boxes.
Validation and test annotations remain unchanged.

Bounding boxes are used for training supervision and localization
evaluation. They are not model inputs during inference.

## Experimental Environment

The experiments were conducted using:

- Windows
- Python 3.12.10
- NVIDIA GeForce RTX 5060
- PyTorch 2.12.1+cu130
- Torchvision 0.27.1+cu130
- CUDA 13.0 PyTorch build
- Automatic mixed precision (FP16)

The complete installed package versions are recorded in
`requirements.txt`.

## Installation

Clone the repository:

```powershell
git clone https://github.com/AlzahraAli/chest-xray-flexible-guidance.git
cd chest-xray-flexible-guidance
```

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the CUDA-enabled PyTorch packages:

```powershell
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
```

Install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

Verify CUDA availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The training and evaluation scripts require a compatible CUDA GPU.

PyTorch installation reference:
https://pytorch.org/get-started/previous-versions/

## Dataset

The experiments use the labelled Stage 2 training portion of
the RSNA Pneumonia Detection Challenge dataset.

RSNA challenge:
https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

Kaggle dataset:
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

NIH Chest X-ray dataset metadata:
https://nihcc.app.box.com/v/ChestXray-NIHCC

The RSNA-to-NIH image mapping is available through the official
RSNA challenge website under the dataset download section.

NIH patient identifiers are used to construct patient-disjoint
splits. NIH disease labels are not used as classification targets.

Place the required files in a directory named `cs421-data`
inside the user's home directory.

Expected directory structure:

```text
cs421-data/
    stage_2_train_images/
        *.dcm
    stage_2_train_labels.csv
    stage_2_detailed_class_info.csv
    Data_Entry_2017_v2020.csv
    pneumonia-challenge-dataset-mappings_2018.json
```

If the downloaded mapping or metadata has a different filename,
rename it to match the expected filename.

The original medical images are not distributed through this
repository. Users must obtain the data from the original sources
and comply with their applicable data-use and attribution terms.

Dataset reference:

G. Shih et al., "Augmenting the National Institutes of Health
Chest Radiograph Dataset with Expert Annotations of Possible
Pneumonia," Radiology: Artificial Intelligence, 2019.
https://doi.org/10.1148/ryai.2019180041

## Data Preparation

Run:

```powershell
python prepare_data.py
```

The script combines repeated bounding-box annotations, verifies
NIH patient identities and constructs patient-disjoint training,
validation and test sets using split seed 42.

Dataset statistics:

| Split | Images | Patients | Positive | Negative |
|---|---:|---:|---:|---:|
| Training | 21456 | 9161 | 4885 | 16571 |
| Validation | 2628 | 1145 | 585 | 2043 |
| Test | 2600 | 1146 | 542 | 2058 |
| Total | 26684 | 11452 | 6012 | 20672 |

Patient overlap between splits is zero.

The generated split files are stored in `cs421-data/splits/`.
Their original SHA-256 checksums are recorded in
`evidence/split_hashes.csv`.

The evaluation scripts verify the split files against these
checksums. If verification fails, investigate the dataset
versions, ordering, software environment and CSV line endings
before proceeding. Do not replace the reference hashes merely
to bypass a failed check.

Images are resized to 320 x 320 pixels, replicated into three
channels and normalized using ImageNet statistics.

No geometric image augmentation is applied.

Test the dataset loader and model:

```powershell
python dataset.py
python model.py
```

## Training Configuration

| Parameter | Value |
|---|---|
| Backbone | ImageNet-pretrained DenseNet-121 |
| Input resolution | 320 x 320 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 16 |
| Maximum epochs | 5 |
| Early stopping patience | 5 |
| Training seeds | 42, 43, 44 |
| Classification loss | Binary cross-entropy with logits |
| Spatial temperature | 1.0 |
| Distance parameter (rho) | 0.05 |
| Selected lambda | 1.0 |
| Checkpoint selection | Highest validation AUROC |

The original Phase I maximum of 20 epochs was reduced to five
under a fixed experimental training budget. The same limit was
applied to all methods. Convergence within this budget is not
guaranteed.

The classification-only model is trained once per seed and
reused across annotation conditions because its objective
does not depend on bounding boxes.

## Annotation Perturbations

- Unchanged: original bounding boxes.
- Translate: random horizontal and vertical shifts of up to
  10% of each box's width and height.
- Shrink: centered reduction of width and height by 20%.
- Enlarge: centered increase of width and height by 20%.

All transformed boxes are clipped to image boundaries.

The FixedTrainingBoxes class generates and caches training
annotations using image-specific seeds. Modified annotations
are fixed across epochs and shared across methods within the
same experimental condition and seed.

The additional perturb_dataset_boxes helper is not used by
the experimental training pipeline.

## Reproducing the Experiments

The commands below are intended for a fresh experimental
environment. Existing training directories are not overwritten.

### Feasibility tests

```powershell
python train.py --feasibility

python train_guided.py --method distance_weighted --condition translate --lam 1.0 --seed 42 --feasibility
```

### Lambda selection

Train both candidate lambda values using unchanged annotations
and seed 42.

```powershell
$methods = @("binary", "fixed_padding", "distance_weighted")
$lambdas = @("0.1", "1.0")

foreach ($method in $methods) {
    foreach ($lam in $lambdas) {
        python -u train_guided.py --method $method --condition unchanged --lam $lam --seed 42 --epochs 5

        if ($LASTEXITCODE -ne 0) {
            throw "Lambda-selection run failed."
        }
    }
}
```

Select lambda using the highest validation AUROC obtained
during each run.

All three guidance methods selected lambda = 1.0.

### Primary experiments

Execute the remaining primary experiments. The three unchanged
seed-42 models with lambda = 1.0 have already been trained
during hyperparameter selection and are reused.

```powershell
$methods = @("binary", "fixed_padding", "distance_weighted")
$conditions = @("unchanged", "translate", "shrink", "enlarge")
$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {

    python -u train.py --epochs 5 --seed $seed

    if ($LASTEXITCODE -ne 0) {
        throw "Baseline training failed."
    }

    foreach ($condition in $conditions) {
        foreach ($method in $methods) {

            if ($seed -eq 42 -and $condition -eq "unchanged") {
                continue
            }

            python -u train_guided.py --method $method --condition $condition --lam 1.0 --seed $seed --epochs 5

            if ($LASTEXITCODE -ne 0) {
                throw "Guided training failed."
            }
        }
    }
}
```

The experiment contains 39 primary runs and three additional
lambda-selection runs, giving 42 formal training runs in total.

Training histories and model checkpoints are stored in
`cs421-data/runs/`.

Verify the completed training runs:

```powershell
python verify_training.py
```

## Evaluation

The best checkpoint from each primary run is evaluated.

Classification metrics:

- AUROC
- F1-score

Localization metrics:

- Mean Intersection over Union (IoU)
- Pointing-game hit rate

The classification threshold maximizes validation F1-score
over candidate probabilities from 0.05 to 0.95 in steps of 0.01.

Activation maps are bilinearly upsampled to 320 x 320 and
normalized by their maximum value. The heatmap threshold
maximizes mean validation IoU over candidates from 0.1 to 0.9.

Both thresholds are selected using validation data and held
fixed during test evaluation.

Localization metrics include all annotated positive images,
regardless of classification correctness.

First, verify the validation pipeline:

```powershell
python evaluate.py
python -u run_evaluation.py --check
```

The standalone evaluate.py command uses synthetic data only.

Evaluate the 39 primary models:

```powershell
python -u run_evaluation.py --all
```

Aggregate the results:

```powershell
python -u aggregate_results.py
```

Generate qualitative heatmaps:

```powershell
python -u generate_heatmaps.py
```

Evaluation results are stored in `cs421-data/evaluation/`.
Final tables and figures are stored in
`cs421-data/final_results/`.

Results are summarized using the mean and sample standard
deviation across three seeds. Robustness differences are
calculated relative to each method's unchanged condition.

The classification-only results are reused across conditions
without additional training.

## Repository Structure

```text
prepare_data.py       Patient-disjoint dataset preparation
dataset.py            DICOM preprocessing and data loading
model.py              DenseNet-121 and spatial activation maps
losses.py             Classification and spatial guidance losses
perturbations.py      Bounding-box transformations
train.py              Classification-only training
train_guided.py       Guided training
verify_training.py    Training and checkpoint verification
evaluate.py           Classification and localization metrics
run_evaluation.py     Checkpoint evaluation pipeline
visualize.py          Statistical aggregation and visualization
aggregate_results.py  Final result verification and aggregation
generate_heatmaps.py  Matched qualitative heatmap generation
requirements.txt      Installed software dependencies
evidence/             Experimental records and comparison figures
```

## Experimental Evidence

The evidence directory contains:

- Dataset split statistics and checksums.
- Training histories for all 42 formal runs.
- Configuration records for guided training.
- Evaluation metric records for all 39 primary runs.
- Per-run and aggregated performance tables.
- Paired robustness differences.
- Quantitative comparison figures.
- Figure SHA-256 checksums.

The original checkpoints, individual predictions, medical
images and patient-level records are retained outside the
repository.

Fixed seeds and recorded software versions support
reproducibility. Identical floating-point results across
different hardware and software environments are not guaranteed.

PyTorch reproducibility reference:
https://docs.pytorch.org/docs/stable/notes/randomness.html

## Contributions

Helh Bader Alhussain:
Dataset preparation, preprocessing, model construction,
training pipeline, experimental execution, integration and
verification.

Alzahra Ali Alabdullah:
Spatial guidance losses, annotation perturbations, evaluation
metrics, statistical aggregation and visualization.

Individual implementation contributions are recorded in the
Git commit history.