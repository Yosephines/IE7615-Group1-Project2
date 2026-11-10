# Text-encoder integration

The pretrained Stable Diffusion v1.5 weights remain frozen.
No training or fine-tuning was performed.

Architecture:
Text -> CLIP tokenizer -> CLIP token-level hidden states (1, 77, 768)
-> embedding interpolation -> U-Net cross-attention
-> scheduler denoising steps -> VAE decoding -> image.

Custom conditioning:
h(alpha) = h(empty) + alpha * (h(text) - h(empty)).
The empty-prompt embeddings remain the classifier-free guidance reference.
Alpha 0 and 1 use the original endpoint embeddings directly.

Four validation captions were used for the fixed alpha settings 0, 0.5 and 1.
The 16 original baseline prompts were used only to check compatibility
between the standard pipeline and the explicit integration at alpha 1.
No setting was selected using the evaluation outputs.

All runs use 512x512 resolution, 25 steps, guidance scale 7.5,
and the same seed within each comparison.
The scheduler and software versions are recorded in environment.json.

Pixel MAE measures output compatibility, not image quality.
Visual alignment observations must be completed after inspecting the galleries.
Runtime measurements include text encoding, generation and pipeline processing,
but exclude initial model loading and writing image files.
GPU memory is peak allocated process memory, including resident model weights.

Sources:
https://github.com/Yosephines/IE7615-Group1-Project2
https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
