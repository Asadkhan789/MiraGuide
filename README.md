# MIRA Guide

Code for preparing knowledge-graph evidence and training the connector on ScienceQA, MultiModalQA (MMQA), and VQA-X. The full and smoke entry points use the same model modules; only the number of questions and training epochs differ. Run commands from the repository root in PowerShell.

## Full and smoke runs

After setup and obtaining the dataset annotations, run one dataset at a time:

```powershell
.\run_full.ps1 -Dataset scienceqa
.\run_full.ps1 -Dataset mmqa
.\run_full.ps1 -Dataset vqax
```

For a small end-to-end check of the same pipeline:

```powershell
.\run_smoke.ps1 -Dataset scienceqa
.\run_smoke.ps1 -Dataset mmqa
.\run_smoke.ps1 -Dataset vqax
```

Both entry points prepare captions, GPT-5 triples, subgraphs, and evidence; train the fusion, slot, router, KG-JSA, and optional latent-compressor weights; and evaluate answers and explanations. `run_full.ps1` uses all available labeled train and held-out questions. `run_smoke.ps1` uses 10 training and 5 held-out questions. The GPT-5 stage prompts for a gateway key. Running a smoke MMQA or VQA-X preparation replaces the local `selected_*.jsonl` files; rerun the full entry point to restore full-split selections. These scripts implement this repository's final-layer KG-JSA connector, not every component proposed in the original MIRA architecture.

The standalone `smoke_scienceqa.py` and `smoke_openqa_connector.py` files are diagnostics. Use `run_full.ps1` for full-split training or `run_smoke.ps1` for an end-to-end smoke run of the connector.

## 1. Set up Python and models

Use a CUDA-enabled PyTorch environment with enough GPU memory for Llama 3.1 8B. On the original Windows system, the environment is named `pytorch`:

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python download_models.py
```

The CUDA check must print `True`. Llama access may require a Hugging Face login. `download_models.py` places Llama in `models/llama-3.1-8b-instruct`; the other models use the Hugging Face cache. To use an existing Llama directory, set `$env:MIRA_LLAMA_MODEL = 'C:\path\to\llama-3.1-8b-instruct'` before training.

The `data/`, `models/`, and `outputs/` directories are excluded from Git. If those files already exist on another checkout of this project, copy them into the same directories here instead of downloading them again. Prepared stage files and checkpoints are needed to evaluate a saved connector.

## 2. ScienceQA

Download the [ScienceQA dataset](https://huggingface.co/datasets/derek-thomas/ScienceQA), prepare captions, GPT-5 triples, subgraphs, and evidence, then train and evaluate:

```powershell
python download_scienceqa.py
python prepare_scienceqa_mira_stages.py --kg-model gpt-5 --output outputs/scienceqa_stages
python train_mira_scienceqa.py --prepared outputs/scienceqa_stages --decoder-mode attention --compression --generate-explanations --output outputs/scienceqa_run
Get-Content outputs/scienceqa_run/summary.json
```

The stage command prompts for a GPT-5 gateway key without saving it. Local Qwen2.5-VL generates image captions; it does not use a Qwen API. The commands use the available train and test splits by default. To prepare a smaller trial, pass the **same** `--train-count` and `--test-count` values to both the stage and training commands.

To evaluate a saved connector again on the same prepared split:

```powershell
python train_mira_scienceqa.py --prepared outputs/scienceqa_stages --checkpoint outputs/scienceqa_run/mira_connector.pt --decoder-mode attention --compression --generate-explanations --output outputs/scienceqa_eval
```

To train the optional latent prefix compressor and evaluate both memory settings on the same ScienceQA questions:

```powershell
python train_mira_scienceqa.py --prepared outputs/scienceqa_stages --decoder-mode attention --compression --generate-explanations --output outputs/scienceqa_compressed
```

The default compressor budgets are `--compression-m0 16`, `--compression-r-qa 16`, and `--compression-r-exp 8`; each can be changed on the command line. The compressor receives answer-loss gradients during training. Explanation generation uses the shared trained weights with the explanation budget. `predictions.json` includes compressed and uncompressed answers from the same connector checkpoint; the uncompressed arm is an inference ablation, not a separately trained model. To use a smaller prepared split, pass matching `--train-count` and `--test-count` values.

## 3. MultiModalQA (MMQA)

Download the [official MMQA train/dev data](https://github.com/allenai/multimodalqa), prepare **every labeled train and dev question** by default, and extract all images referenced by those questions:

```powershell
python download_mmqa.py
python prepare_mmqa_dataset.py
python extract_mmqa_images.py
```

The image archive is large. To fetch just the images referenced by the prepared questions, use `python download_mmqa.py --skip-image-archive`, then `python prepare_mmqa_dataset.py` and `python extract_mmqa_images_remote.py` in place of the three commands above. When the full splits are prepared, this fetches the images needed for those full splits. Then prepare evidence and train:

```powershell
python prepare_openqa_mira_stages.py --dataset mmqa --output outputs/mmqa_stages
python train_mira_openqa.py --dataset mmqa --prepared outputs/mmqa_stages --compression --output outputs/mmqa_run
Get-Content outputs/mmqa_run/summary.json
```

The stage command prompts for the GPT-5 gateway key. `prepare_mmqa_dataset.py` writes the full labeled train/dev splits by default; `--train-count` and `--val-count` can select fewer records. For a matching smaller connector run, add `--train-limit` and `--val-limit` to `train_mira_openqa.py`.

To evaluate a saved connector on the same selected records:

```powershell
python train_mira_openqa.py --dataset mmqa --prepared outputs/mmqa_stages --checkpoint outputs/mmqa_run/mira_connector.pt --compression --output outputs/mmqa_eval
```

## 4. VQA-X

Download the [VQA-X train/val annotations](https://github.com/fawazsammani/nlxgpt#annotations-download) and place `vqaX_train.json` and `vqaX_val.json` in `data/vqax/`. The preparer processes **every annotated train and validation question** and downloads all referenced COCO images by default. The stage and training commands below also use all prepared records by default:

```powershell
python prepare_vqax_dataset.py
python prepare_openqa_mira_stages.py --dataset vqax --output outputs/vqax_stages
python train_mira_openqa.py --dataset vqax --prepared outputs/vqax_stages --compression --output outputs/vqax_run
Get-Content outputs/vqax_run/summary.json
```

The preparation step writes normalized `selected_train.jsonl` and `selected_val.jsonl`; despite those filenames, they contain the full splits when no count flags are passed. Optional `--train-count` and `--val-count` flags are for smaller trials. Dataset files and generated outputs stay outside Git through `.gitignore`. The stage command prompts for the GPT-5 gateway key. To evaluate the saved connector again:

```powershell
python train_mira_openqa.py --dataset vqax --prepared outputs/vqax_stages --checkpoint outputs/vqax_run/mira_connector.pt --compression --output outputs/vqax_eval
```

## Output files

Stage folders contain `00_input.json`, `01_caption.json`, `02_kg.json`, `03_subgraph.json`, and `04_evidence.json` for each question. Training writes `mira_connector.pt`, `split.json`, predictions, and `summary.json` under the chosen `--output` directory. These runtime files remain local and are not committed to Git.
