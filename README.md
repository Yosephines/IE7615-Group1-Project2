# IE 7615 - Group 1 - Project 2

Text-to-image generation with a Transformer text encoder and Stable Diffusion.

## Start here

- **Milestone 1:** Read the [Milestone 1 report](reports/milestone01_report.pdf). It summarizes the dataset, the CLIP embedding pipeline, the baseline generation and the plan. The other Milestone 1 documents are the [project proposal](reports/milestone01_proposal.pdf), the [dataset memo](reports/milestone01_dataset_memo.pdf) and the [baseline gallery and analysis](reports/milestone01_baseline_gallery_memo.pdf).

| Milestone 1 deliverable | File | Owner |
| --- | --- | --- |
| Dataset-preparation notebook | [`notebooks/01_dataset_preparation.ipynb`](notebooks/01_dataset_preparation.ipynb) | Yosephine Tong |
| CLIP text/image embedding notebook | [`notebooks/02_clip_embeddings.ipynb`](notebooks/02_clip_embeddings.ipynb) | Dario Garza |
| Baseline Stable Diffusion notebook | [`notebooks/03_stable_diffusion_generation.ipynb`](notebooks/03_stable_diffusion_generation.ipynb) | Aditi Rao |
| Dataset memo | [`reports/milestone01_dataset_memo.pdf`](reports/milestone01_dataset_memo.pdf) | Yosephine Tong |
| Baseline gallery and analysis | [`reports/milestone01_baseline_gallery_memo.pdf`](reports/milestone01_baseline_gallery_memo.pdf) ([full comparison sheet](reports/baseline_comparisons.png)) | Aditi Rao |
| Project proposal | [`reports/milestone01_proposal.pdf`](reports/milestone01_proposal.pdf) | Murat Akca |
| Milestone 1 report | [`reports/milestone01_report.pdf`](reports/milestone01_report.pdf) | Murat Akca |

## Setup

**Local (dataset notebook and reports).** Verified with Python 3.13.9. From the repo folder:

```sh
python3.13 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ie7615-project2 --display-name "IE7615 Project 2"
```

Open the dataset notebook with this kernel and run all cells. The first run downloads the COCO annotations and the selected images; they stay local and are excluded from Git.

**Google Colab (CLIP and Stable Diffusion notebooks).** Open notebook 02 or 03 from GitHub in Colab (*File → Open notebook → GitHub*, or the "Open in Colab" badge in notebook 02), select a **T4 GPU** runtime and run all cells. Each notebook installs its Hugging Face packages in its first cell. Notebook 02 pins Transformers 5.20.1, Diffusers 0.41.0 and Accelerate 1.15.0 and reads `data/manifest.csv` from this repository. Notebook 03 reads `baseline_prompts.csv` and `manifest.csv` from the Colab working folder, so upload the two files from `data/` first. Model weights download from Hugging Face on the first run.

## Reproduce the reported numbers

| Result | How to rerun | Compare with |
| --- | --- | --- |
| Dataset: 1,000 pairs, splits, review counts | Run the [dataset notebook](notebooks/01_dataset_preparation.ipynb) locally. It rebuilds the subset and asserts that the manifest hash is unchanged. | [`data/manifest.csv`](data/manifest.csv), [dataset memo](reports/milestone01_dataset_memo.pdf) |
| CLIP pipeline: 7/8 and 30/32 image→text top-1 | Run the [CLIP notebook](notebooks/02_clip_embeddings.ipynb) on a Colab T4 | Saved outputs in the notebook, report Section 3.2 |
| Baseline gallery: 16 prompts, seeds 42–57, 25 steps | Run the [Stable Diffusion notebook](notebooks/03_stable_diffusion_generation.ipynb) on a Colab T4 | [`reports/baseline_comparisons.png`](reports/baseline_comparisons.png) |
| Report PDFs | `python reports/src/make_baseline_gallery.py` then `sh reports/build_reports.sh` (macOS; needs pandoc and Google Chrome) | [`reports/`](reports/) |

Images generated on different GPUs or library versions can differ from the committed gallery even with the same seed.

## Folders

| Folder | Contents |
| --- | --- |
| `data/` | Dataset manifest (one row per pair with caption, split, source and license) and the 16 fixed baseline prompts |
| `notebooks/` | Milestone 1 notebooks: dataset preparation, CLIP embeddings and baseline generation |
| `project/` | Dataset preparation and checking scripts, caption choices, split assignments and review records |
| `reports/` | Milestone 1 PDFs, the baseline comparison sheet, and the report sources and build script in `reports/src/` |

## Milestone 1 results

| Part | Result |
| --- | --- |
| Dataset | 1,000 COCO 2017 image–caption pairs, 512 × 512, all 80 categories; 800 development / 100 validation / 100 evaluation, split by image |
| CLIP ViT-B/32 embeddings | 512-dimensional text and image embeddings; image→text top-1 of 30/32 (93.75%) on validation pairs |
| Stable Diffusion v1.5 baseline | 16 prompts generated; main subjects reliable, object counts, spatial relations and specific details weak |

The 16 baseline prompts come from the evaluation split. They were inspected only qualitatively with the default pretrained model, and no setting was chosen from them. From Milestone 2 on, settings are selected on validation prompts and the evaluation split is used once at the end.

## Sources and licenses

Image sources and licenses are recorded in the manifest and in the [third-party notices](THIRD_PARTY_NOTICES.md). COCO annotations are CC BY 4.0; images keep their Flickr CC BY 2.0 or CC BY-SA 2.0 licenses. Stable Diffusion v1.5 is used under the CreativeML OpenRAIL-M license.
