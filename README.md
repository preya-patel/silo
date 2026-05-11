# SILO+: Image Restoration via Attention and Multi-Scale Learning

CS/ECE 8690 Computer Vision — Final Project  
**Preya Patel and Mithilesh Gollapelli**  
University of Missouri-Columbia

## What's in this folder

| Item | Description |
|---|---|
| `REPORT.pdf` | Full paper in IEEE format |
| `PRESENTATION.pdf` | Slide deck used for the in-class presentation |
| `train_and_eval.ipynb` | Colab notebook that runs the full pipeline end-to-end |
| `code/silo_v1.py` | SILO baseline implementation (vanilla CNN operator) |
| `code/silo_v2.py` | SILO+ implementation (multi-scale + spatial attention) |
| `code/requirements.txt` | Python dependencies |
| `checkpoints/` | Trained operator weights for both tasks and both methods |
| `data/test_images/` | 5 FFHQ test images used for evaluation |
| `results/` | Reconstruction grids, bar charts, training curves, and metric CSVs |

## What we built

We reproduced SILO (Raphaeli, Man, Elad, ICCV 2025) as our baseline (v1) and proposed SILO+ (v2), which adds two enhancements to the latent operator:

- A **multi-scale block** with parallel 3×3, 5×5, 7×7 convolutions
- A **spatial attention module** that learns per-position importance

Under identical training and sampling conditions on FFHQ, SILO+ outperforms the SILO baseline on every metric on both inpainting and Gaussian deblurring.

## Reproducing the results

We ran everything on Google Colab with a A100 GPU. The cleanest path to reproduce is to open `train_and_eval.ipynb` in Colab and run cells top to bottom. The notebook handles dependency installation, drive mounting, training, evaluation, and figure generation.

For command-line use:

1. Install dependencies:
```
   pip install -r code/requirements.txt
```

2. Evaluate with our trained checkpoints (about 25 minutes per task on T4):
```
   python code/silo_v2.py --mode eval \
       --task inpaint \
       --ckpt checkpoints/silo_v2_inpaint.pt \
       --test_dir data/test_images \
       --num_eval 5 --sample_steps 500 --eta 2.0 \
       --out_dir results/silo_v2_eval_inpaint
```

3. To retrain from scratch, replace `--mode eval` with `--mode train` and point `--train_dir` at your FFHQ training images (we used the first 1000 of FFHQ). 

Swap `--task inpaint` for `--task gauss_blur` to run the deblurring task, and `silo_v2.py` for `silo_v1.py` to run the baseline.

## Hardware and runtime

- GPU: Google Colab A100
- Training: ~30 minutes per task (2000 steps, batch size 4, 512×512)
- Evaluation (5 images): ~25 minutes per task at 500 sampling steps

## Headline results

| Task | Metric | Measurement | SILO v1 | SILO+ v2 |
|---|---|---|---|---|
| Inpaint | PSNR ↑ | 24.13 | 26.33 | **26.96** |
| Inpaint | LPIPS ↓ | 0.0810 | 0.1295 | **0.0669** |
| Blur | PSNR ↑ | 27.17 | 26.30 | **26.46** |
| Blur | LPIPS ↓ | 0.3711 | 0.2167 | **0.1985** |

SILO+ wins on every metric on both tasks. The largest gains are perceptual: 48% LPIPS reduction on inpainting and 47% LPIPS reduction on deblurring relative to the degraded measurement.

## Team contributions

Both authors contributed equally. Preya set up the training and evaluation pipeline on Colab, implemented the SILO baseline, and produced the quantitative results and plots. Mithilesh designed and built the SILO+ operator with the multi-scale block and spatial attention, handled sampling-side tuning, and prepared the qualitative reconstruction grids. The report and slides were written together.

## Acknowledgments

Built on top of the SILO algorithm (Raphaeli et al., ICCV 2025) and the Hugging Face Diffusers library. Stable Diffusion weights from the Realistic Vision v5.1 checkpoint on Hugging Face Hub.
