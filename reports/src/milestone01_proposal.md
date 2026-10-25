---
title: Initial Project Proposal
---

<div class="doc-head">
<div class="course">IE 7615 Deep Learning for AI · Group 1 · Text-to-Image Generation Using Transformer and Diffusion Models</div>
<h1>Milestone 1: Initial Project Proposal</h1>
<div class="authors">Murat Akca · Aditi Rao · Dario Garza · Yosephine Tong</div>
</div>

## 1. Goal

We will build and evaluate a text-to-image pipeline in which a Transformer text encoder conditions a pretrained latent diffusion model. The project studies how sampling and conditioning settings change semantic alignment with the prompt, image quality and output diversity. The Milestone 1 baseline with Stable Diffusion v1.5 rendered the main subject reliably but struggled with object counts, spatial relations and specific details. Those failure modes guide the experiments below.

## 2. Dataset

We use a curated subset of **COCO 2017 Captions**: 1,000 image–caption pairs from the validation images, one caption per image. COCO's everyday multi-object scenes test exactly the counting and layout weaknesses seen in the baseline. Its object annotations show which categories each prompt covers, and it is the standard reference set for text-to-image FID. Flickr30k is centered on people and has more restrictive terms. Conceptual Captions and LAION have noisier captions and are harder to screen for licensing and content.

The subset keeps CC BY and CC BY-SA images, cleaned and deduplicated captions of 5–35 words, and 512 × 512 RGB images padded without distortion. All pairs were reviewed visually. They are split by image into 800 development, 100 validation and 100 evaluation pairs. **Development** images are the real-image reference for FID and KID. **Validation** captions are used for every setting we choose, through a fixed 16-prompt tuning set plus all 100 captions. **Evaluation** captions are held out for one final run. The 16 Milestone 1 baseline prompts come from the evaluation split. They were inspected only with the default pretrained model, and no setting was chosen from them.

## 3. Model and conditioning approach

**Model.** We will use pretrained Stable Diffusion v1.5 through Hugging Face Diffusers, pinned to a fixed model revision. The VAE maps a 512 × 512 image to a 4 × 64 × 64 latent, where diffusion runs. A U-Net predicts the noise at each step and reads the prompt through cross-attention. The frozen CLIP ViT-L/14 text Transformer encodes the prompt as 77 token embeddings of size 768. Classifier-free guidance combines conditional and unconditional (empty-prompt) predictions; the guidance scale sets how strongly the text steers sampling. The CLIP ViT-B/32 model from the Milestone 1 embedding notebook scores image–text alignment. Because it is not the encoder that conditions generation, the evaluation is less circular.

**Milestone 2 (end of Module 10).** We will compute the prompt embeddings ourselves and pass them to the diffusion model, making the text→embedding→image path explicit. On the validation tuning set, with three seeds per prompt, we will vary the guidance scale (1, 3, 5, 7.5, 10, 15), sampling steps (10, 25, 50), sampler and noise schedule (PNDM default, DDIM, DPM-Solver++, Euler, timestep spacing), and conditioning strength by interpolating between the unconditional and conditional embeddings (α = 0.25, 0.5, 0.75, 1.0). Each setting is scored with CLIP similarity and a checklist built from the baseline failure modes: main subject, object counts, spatial relations and stated attributes.

**Milestone 3 (end of Module 12).** We will compute FID, KID and Inception Score, and measure diversity as the similarity between images from the same prompt with different seeds. This supports the fidelity–diversity analysis. A sensitivity study on validation fixes the final settings; the evaluation split is then used once to compare the default and selected settings. A LoRA fine-tune on the development pairs is an optional extension if time allows.

**Compute.** Experiments and the offline live demo will run on a local Apple Silicon laptop (M1 Max, Metal backend), with Google Colab T4 GPUs as a second environment. Package versions, model revisions and seeds are recorded with every result.

## 4. Team roles for the generative work

Roles are initial and may be adjusted as the work develops.

| Member | Role |
|------|----------------------|
| Murat Akca | Pipeline integration, repository and reproducibility, offline live demo, milestone report integration |
| Aditi Rao | Generation experiments: samplers, sampling steps and guidance scale |
| Dario Garza | Text conditioning and sensitivity sweeps, CLIP-based alignment scoring |
| Yosephine Tong | Evaluation metrics (FID, KID, IS), documentation and the final IEEE report |

## 5. Anticipated risks

| Risk | Mitigation |
|---|---|
| **Compute budget:** a grid with several seeds means hundreds of images per milestone. | 16-prompt tuning set, small grid, half precision on GPU, overnight local runs, Colab as fallback. |
| **Evaluation feasibility:** FID is biased at a few hundred images; CLIP scores can reward keywords over correct scenes. | KID alongside FID, equal sample sizes, an independent CLIP model and the qualitative checklist. |
| **Licensing:** COCO annotations are CC BY 4.0, images keep their Flickr CC BY/BY-SA licenses, and SD v1.5 carries CreativeML OpenRAIL-M use restrictions. | Per-image source and license in the manifest, attribution beside reference images, images kept out of Git, safety checker on. |
| **Train–test overlap:** COCO images may be in the web data that pretrained Stable Diffusion and CLIP. | Cannot be ruled out; stated as a limitation, and settings are compared relative to each other. |
| **Reproducibility:** the same seed gives different images on different GPUs. | Pinned versions and model revisions, fixed seeds, saved configurations and committed results. |
| **Bias and misuse:** COCO over-represents people and common objects; generated people may reflect stereotypes. | Bias discussed in the analysis; the demo is limited to project prompts with the safety filter on. |
