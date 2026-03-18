import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = Path(__file__).resolve().parent
FIGS_DIR = REPORT_DIR / "figs"

CNN_INTRO = ROOT / "02_CNN_Introduction.ipynb"
NET_TRAIN = ROOT / "03_Network_Training.ipynb"
RNN = ROOT / "05_RNN.ipynb"
YOLO = ROOT / "10_Data_Classification_YOLO.ipynb"


def _load_nb(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell):
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else str(src)


def _cell_text(cell):
    text = ""
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            text += "".join(output.get("text", []))
        if output.get("output_type") == "execute_result":
            text += str(output.get("data", {}).get("text/plain", ""))
    return text


def _cell_images(cell):
    images = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            images.append(data["image/png"])
    return images


def _write_png(b64_data: str, path: Path):
    png_bytes = base64.b64decode(b64_data)
    path.write_bytes(png_bytes)


def export_cnn_intro():
    nb = _load_nb(CNN_INTRO)
    bar_png = None
    reg_png = None

    for cell in nb.get("cells", []):
        text = _cell_text(cell)
        images = _cell_images(cell)
        if "Architecture results (best val acc)" in text and images:
            bar_png = images[0]
        if "Architecture results (best validation MAPE)" in text and images:
            reg_png = images[0]

    if bar_png:
        _write_png(bar_png, FIGS_DIR / "cnn_task1_bar.png")
    else:
        raise RuntimeError("Could not find Task 1 bar plot in 02_CNN_Introduction.ipynb.")

    if reg_png:
        _write_png(reg_png, FIGS_DIR / "cnn_task2_loss.png")
    else:
        raise RuntimeError("Could not find Task 2 loss plot in 02_CNN_Introduction.ipynb.")


def export_net_train():
    nb = _load_nb(NET_TRAIN)
    summary_cell_images = None

    for cell in nb.get("cells", []):
        text = _cell_text(cell)
        images = _cell_images(cell)
        if "Summary table (best validation accuracy)" in text and images:
            summary_cell_images = images
            break

    if not summary_cell_images or len(summary_cell_images) < 2:
        raise RuntimeError("Could not find augmentation/SGD plots in 03_Network_Training.ipynb.")

    _write_png(summary_cell_images[0], FIGS_DIR / "net_aug_loss.png")
    _write_png(summary_cell_images[1], FIGS_DIR / "net_sgd_loss.png")


def export_arch_sketch():
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 2.5))
    ax = plt.gca()
    ax.axis("off")

    blocks = [
        "Input\n32x32x3",
        "Conv 32\nReLU\nMaxPool",
        "Conv 64\nReLU\nMaxPool",
        "Conv 128\nReLU\nMaxPool",
        "GlobalAvgPool",
        "FC 10",
    ]

    x = 0.02
    for i, label in enumerate(blocks):
        ax.add_patch(plt.Rectangle((x, 0.25), 0.14, 0.5, fill=False, linewidth=1.2))
        ax.text(x + 0.07, 0.5, label, ha="center", va="center", fontsize=7)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 0.16, 0.5), xytext=(x + 0.14, 0.5),
                        arrowprops=dict(arrowstyle="->", linewidth=1))
        x += 0.16

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "cnn_task1_arch.png", dpi=200)
    plt.close()


def export_rnn():
    nb = _load_nb(RNN)
    task1_img = task2_img = task3_img = None

    for cell in nb.get("cells", []):
        src = _cell_source(cell)
        images = _cell_images(cell)
        if not images:
            continue
        if "Coursework summary: Task 1" in src or "window_size sweep" in src:
            task1_img = images[-1]
        if "Coursework summary: Task 2" in src or "accuracy curves" in src:
            task2_img = images[-1]
        if "Coursework summary: BLEU" in src or "BLEU vs temperature" in src:
            task3_img = images[-1]

    if task1_img:
        _write_png(task1_img, FIGS_DIR / "rnn_task1_curves.png")
        print("  rnn_task1_curves.png")
    else:
        print("  [skip] rnn_task1_curves.png (no output in 05_RNN)")
    if task2_img:
        _write_png(task2_img, FIGS_DIR / "rnn_task2_curves.png")
        print("  rnn_task2_curves.png")
    else:
        print("  [skip] rnn_task2_curves.png (no output in 05_RNN)")
    if task3_img:
        _write_png(task3_img, FIGS_DIR / "rnn_task3_bleu.png")
        print("  rnn_task3_bleu.png")
    else:
        print("  [skip] rnn_task3_bleu.png (no output in 05_RNN)")


def export_yolo():
    nb = _load_nb(YOLO)
    curves_img = None

    for cell in nb.get("cells", []):
        src = _cell_source(cell)
        images = _cell_images(cell)
        if not images:
            continue
        if "Coursework summary" in src and "learning curves" in src:
            curves_img = images[-1]
            break

    if curves_img:
        _write_png(curves_img, FIGS_DIR / "yolo_learning_curves.png")
        print("  yolo_learning_curves.png")
    else:
        print("  [skip] yolo_learning_curves.png (no output in 10_Data_Classification_YOLO)")


def main():
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting 02_CNN_Introduction...")
    export_cnn_intro()
    print("Exporting 03_Network_Training...")
    export_net_train()
    print("Exporting 05_RNN...")
    export_rnn()
    print("Exporting 10_Data_Classification_YOLO...")
    export_yolo()
    print("Exporting cnn_task1_arch (sketch)...")
    export_arch_sketch()
    print("Done. Figures saved to", FIGS_DIR)


if __name__ == "__main__":
    main()
