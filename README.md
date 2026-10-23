# IE 7615 - Group 1 - Project 2

Text-to-image generation with a Transformer text encoder and Stable Diffusion.

## Dataset preparation

The initial contribution contains 1,000 COCO image-caption pairs, an executed dataset notebook and a dataset memo. Review revised 133 captions and replaced two images.

- [Dataset notebook](notebooks/01_dataset_preparation.ipynb)
- [Dataset memo](reports/milestone01_dataset_memo.pdf)

## Setup

Verified with Python 3.13.9. From the repo folder in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ie7615-project2 --display-name "IE7615 Project 2"
```

Open the dataset notebook with this kernel and run all cells. The first run downloads the COCO annotations and selected images; these stay local and are excluded from Git.

`data/` contains the manifest and fixed prompts. `project/` contains the two required scripts, saved caption choices and review records. Image sources and licenses are recorded in the manifest and [third-party notices](THIRD_PARTY_NOTICES.md).

## Remaining milestone work

GPU/CLIP embeddings, baseline generation and analysis, the proposal, and the main 5-6-page report will be integrated after team review. Yosephine owns dataset preparation and repo setup; the team will decide the remaining responsibilities.
