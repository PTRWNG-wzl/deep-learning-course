## Overleaf report (teacher template)

1. Open the teacher's template: https://www.overleaf.com/read/qvcbrmrdnkvh  
2. Copy the content of `report/overleaf_report_body.tex` into the main document body (or use `\input{overleaf_report_body}` if you add the file to your project).  
3. Upload figures from `report/figs/` to Overleaf (create a `figs/` folder in your project).  
4. Replace any remaining `--` placeholders in tables with your actual results from the notebooks.

## Final report (notebooks 04–10, 4 pages)

`overleaf_report_body.tex` is structured for the **final report** covering tutorials 04–10 only. It includes:
- **Task 4.1/4.2:** Tiny-ImageNet VGG16, ConvNeXt scaling
- **Task 5.1/5.2/5.3:** RNN regression, text embeddings, BLEU
- **Task 6.1/6.2:** Autoencoders, custom loss
- **Task 7.1/7.2:** VAE/GAN, MAE vs cGAN colourisation
- **Task 8.1:** RL (Q-learning reward curves)
- **Task 9.1/9.2:** Diffusion model
- **Task 10.1:** YOLOv8 roof classification

Appendix includes Task 5.2 curves, Task 7.2 qualitative, Task 9.2 training loss.

## Exporting figures

From the repo root:
```
python report/export_report_figs.py
```

Images for **final report** (notebooks 04–10):
- `rnn_task1_curves.png`, `rnn_task2_curves.png`, `rnn_task3_bleu.png` (from 05)
- `yolo_learning_curves.png` (from 10)

**Manual export** (run notebook cells, then save plots as PNG to `figs/`):
- `convnext_scaling.png` — 04: accuracy/MSE vs params
- `ae_denoise_examples.png` — 06: denoised image examples
- `rl_reward_curves.png` — 08: reward vs episodes
- `diffusion_noise_demo.png` — 09: clean, noised, noise

Placeholder 1×1 PNGs exist for the above; replace with real exports before submission.

## Interim draft (notebooks 02–03)

`report/interim_report_draft.tex` covers notebooks 02–03. Run `export_report_figs.py` for `cnn_task1_bar.png`, `cnn_task1_arch.png`, `cnn_task2_loss.png`, `net_aug_loss.png`, `net_sgd_loss.png`.
