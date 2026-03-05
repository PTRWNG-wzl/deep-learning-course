import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = Path(__file__).resolve().parent
FIGS_DIR = REPORT_DIR / "figs"

CNN_INTRO = ROOT / "02_CNN_Introduction.ipynb"
NET_TRAIN = ROOT / "03_Network_Training.ipynb"


def _load_nb(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    export_cnn_intro()
    export_net_train()
    export_arch_sketch()
    print("Exported figures to", FIGS_DIR)


if __name__ == "__main__":
    main()
