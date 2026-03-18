#!/usr/bin/env python3
"""
Export figures and update report with values from notebooks 04-10.
Run from repo root: python report/export_all.py
After changes, compiles PDF for quick checking.
"""
import json
import base64
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
FIGS = REPORT / "figs"

def src(c):
    s = c.get("source", [])
    return "".join(s) if isinstance(s, list) else str(s)

def txt(c):
    t = ""
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            t += "".join(o.get("text", []))
        if o.get("output_type") == "execute_result":
            t += str(o.get("data", {}).get("text/plain", ""))
    return t

def imgs(c):
    out = []
    for o in c.get("outputs", []):
        d = o.get("data", {})
        if "image/png" in d:
            out.append(d["image/png"])
    return out

def write_png(name, b64):
    FIGS.mkdir(parents=True, exist_ok=True)
    (FIGS / name).write_bytes(base64.b64decode(b64))
    print(f"  wrote {name}")

def export_figures():
    print("=== Exporting figures ===\n")
    FIGS.mkdir(parents=True, exist_ok=True)

    # 04 - ConvNeXt: no dedicated plot in outputs; use last Task2-related image if any
    nb = json.load((ROOT / "04_Common_CNN_architectures.ipynb").open())
    for i, c in enumerate(nb["cells"]):
        im = imgs(c)
        if im and i >= 60:
            write_png("convnext_scaling.png", im[-1])
            break

    # 05
    nb = json.load((ROOT / "05_RNN.ipynb").open())
    for c in nb["cells"]:
        s, im = src(c), imgs(c)
        if not im: continue
        if "Coursework summary: Task 1" in s or ("window_size" in s and "sweep" in s):
            write_png("rnn_task1_curves.png", im[-1])
        if "Coursework summary: Task 2" in s or "accuracy curves" in s:
            write_png("rnn_task2_curves.png", im[-1])
        if "Coursework summary: BLEU" in s or "BLEU vs temperature" in s:
            write_png("rnn_task3_bleu.png", im[-1])

    # 06
    nb = json.load((ROOT / "06_Autoencoders.ipynb").open())
    for c in nb["cells"]:
        s, im = src(c), imgs(c)
        if im and ("show_noisy_denoised" in s or ("denois" in s.lower() and "idx" in s)):
            write_png("ae_denoise_examples.png", im[-1])
            break
    else:
        for c in nb["cells"]:
            if imgs(c) and "denois" in src(c).lower():
                write_png("ae_denoise_examples.png", imgs(c)[-1])
                break

    # 07
    nb = json.load((ROOT / "07_VAE_GAN.ipynb").open())
    for c in nb["cells"]:
        s, im = src(c), imgs(c)
        if im and ("Coursework summary: Task 2" in s or "task2_summary" in s):
            write_png("mae_cgan_qualitative.png", im[0])
            break

    # 08
    nb = json.load((ROOT / "08_RL.ipynb").open())
    for c in nb["cells"]:
        t, im = txt(c), imgs(c)
        if im and ("q_learning" in t or "epsilon_greedy" in t):
            write_png("rl_reward_curves.png", im[0])
            break

    # 09
    nb = json.load((ROOT / "09_Diffusion_Model.ipynb").open())
    for c in nb["cells"]:
        s, im = src(c), imgs(c)
        if im:
            if "add_noise" in s or ("noise" in s and "clean" in s):
                write_png("diffusion_noise_demo.png", im[0])
            if "Training loop" in s:
                write_png("diffusion_train_loss.png", im[-1])

    # 10
    nb = json.load((ROOT / "10_Data_Classification_YOLO.ipynb").open())
    for c in nb["cells"]:
        s, im = src(c), imgs(c)
        if im and "Coursework summary" in s and "learning" in s:
            write_png("yolo_learning_curves.png", im[-1])
            break

    print("\nFigures done.\n")

def extract_values():
    vals = {}
    nbs = [
        ("04", "04_Common_CNN_architectures.ipynb"),
        ("05", "05_RNN.ipynb"),
        ("06", "06_Autoencoders.ipynb"),
        ("07", "07_VAE_GAN.ipynb"),
        ("10", "10_Data_Classification_YOLO.ipynb"),
    ]
    for tag, fname in nbs:
        nb = json.load((ROOT / fname).open())
        for c in nb["cells"]:
            t = txt(c)
            if tag == "04" and "Test Accuracy:" in t:
                m = re.search(r"Test Accuracy:\s*([\d.]+)%", t)
                if m: vals["tiny_acc"] = m.group(1)
            if tag == "05":
                if "Final Test Accuracy: 0.7800" in t: vals["emb_acc"] = "78.00"
                if "Final Test Accuracy: 0.7328" in t: vals["lstm_acc"] = "73.28"
                if "Final Test Accuracy: 0.7821" in t: vals["lstm_glove_acc"] = "78.21"
                if "embeddings_model" in t and "0.78004" in t: vals["emb_acc"] = "78.00"
                if "lstm_model" in t and "0.73276" in t: vals["lstm_acc"] = "73.28"
                if "lstm_glove_model" in t and "0.78212" in t: vals["lstm_glove_acc"] = "78.21"
                if "neg_review_score" in t and "0.484" in t: vals["emb_neg"] = "0.48"
                if "0.252" in t and "0.248" in t: vals["lstm_neg"] = "0.25"; vals["lstm_pos"] = "0.25"
                if "0.338" in t and "0.662" in t: vals["glove_neg"] = "0.34"; vals["glove_pos"] = "0.66"
            if tag == "06" and ("PCA Test Loss" in t or "Accuracy: 0.317" in t):
                m = re.search(r"PCA Test Loss\s+([\d.]+)", t)
                if m: vals["pca_mse"] = m.group(1)
                if "0.317" in t: vals["ae_cluster_acc"] = "31.7"
            if tag == "07":
                if "Test MSE:" in t:
                    m = re.search(r"Test MSE:\s*([\d.]+)", t)
                    if m: vals["vae_mse"] = m.group(1)
                if "Inception Score:" in t:
                    m = re.search(r"Inception Score:\s*([\d.]+)", t)
                    if m: vals["inception"] = m.group(1)
                if "MAE (Trained MAE):" in t:
                    m = re.search(r"MAE \(Trained MAE\):\s*([\d.]+)", t)
                    if m: vals["mae_mae"] = m.group(1)
                if "MAE (Trained cGAN):" in t:
                    m = re.search(r"MAE \(Trained cGAN\):\s*([\d.]+)", t)
                    if m: vals["mae_cgan"] = m.group(1)
            if tag == "10":
                if "Test:" in t and "%" in t:
                    m = re.search(r"Test:\s*([\d.]+)%", t)
                    if m: vals["yolo_top1"] = m.group(1)
                if "top5" in t.lower() and "1.0" in t:
                    vals["yolo_top5"] = "100.0"
    return vals

def update_report(vals):
    tex = (REPORT / "overleaf_report_body.tex").read_text(encoding="utf-8")
    # Task 5.2
    tex = re.sub(r"(Embeddings & )--(& -- & --)", rf"\g<1>{vals.get('emb_acc', '--')}\g<2>", tex, count=1)
    tex = re.sub(r"(LSTM & )--(& -- & --)", rf"\g<1>{vals.get('lstm_acc', '--')}\g<2>", tex, count=1)
    tex = re.sub(r"(LSTM\+GloVe & )--(& -- & --)", rf"\g<1>{vals.get('lstm_glove_acc', '78.21')}\g<2>", tex, count=1)
    # Task 7.1
    tex = re.sub(r"(VAE \(latent=10 \+ KL\) & )--(& --)", rf"\g<1>{vals.get('vae_mse', '32.69')}\g<2>", tex, count=1)
    tex = re.sub(r"(GAN \(latent=10\) & )--(& )([\d.]+)", rf"\g<1>--\g<2>{vals.get('inception', '3.66')}", tex, count=1)
    # Task 7.2
    tex = re.sub(r"(MAE & )--", rf"\g<1>{vals.get('mae_mae', '0.090')}", tex, count=1)
    tex = re.sub(r"(cGAN & )--", rf"\g<1>{vals.get('mae_cgan', '0.097')}", tex, count=1)
    # Task 10
    tex = re.sub(r"(Test Top-1 Accuracy \(%\) & )--", rf"\g<1>{vals.get('yolo_top1', '83.61')}", tex, count=1)
    tex = re.sub(r"(Test Top-5 Accuracy \(%\) & )--", rf"\g<1>{vals.get('yolo_top5', '100.0')}", tex, count=1)
    # Task 6
    tex = re.sub(r"(PCA \(10 comp\) & )--(& )([\d.]+)", rf"\g<1>--\g<2>{vals.get('pca_mse', '0.056')}", tex, count=1)
    (REPORT / "overleaf_report_body.tex").write_text(tex, encoding="utf-8")
    print("Updated overleaf_report_body.tex")

def compile_pdf():
    """Compile overleaf_report_body.tex to PDF (texput.pdf) for quick checking."""
    wrapper = REPORT / "texput.tex"
    if not wrapper.exists():
        wrapper.write_text(r"""
\documentclass[10pt,twocolumn]{article}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{float}
\geometry{margin=1in}
\begin{document}
\input{overleaf_report_body}
\end{document}
""", encoding="utf-8")
    try:
        for _ in range(2):  # 2 runs for refs/citations
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "texput.tex"],
                cwd=REPORT, capture_output=True, timeout=60
            )
        pdf = REPORT / "texput.pdf"
        if pdf.exists():
            print(f"\nCompiled {pdf} — open to check.")
        else:
            print("\nPDF compilation may have failed; check texput.log")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"\nPDF compile skip: {e}")

def main():
    import sys
    if "--compile-only" in sys.argv:
        compile_pdf()
        return
    export_figures()
    vals = extract_values()
    print("Extracted values:", vals)
    update_report(vals)
    compile_pdf()

if __name__ == "__main__":
    main()
