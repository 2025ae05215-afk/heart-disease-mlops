"""EDA: histograms, correlation heatmap, class balance -> screenshots/eda_*.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import NUMERIC, ROOT, TARGET  # noqa: E402
from .data import clean, load_raw  # noqa: E402

OUT = ROOT / "screenshots"


def run() -> dict:
    OUT.mkdir(exist_ok=True)
    df = clean(load_raw())
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax,
                     palette="Set1")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by target")
    fig.tight_layout()
    fig.savefig(OUT / "eda_histograms.png", dpi=130)
    plt.close(fig)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "eda_correlation.png", dpi=130)
    plt.close()

    plt.figure(figsize=(4.5, 4))
    ax = sns.countplot(x=TARGET, hue=TARGET, data=df, palette="Set1",
                       legend=False)
    ax.bar_label(ax.containers[0])
    plt.title("Class balance (1 = disease)")
    plt.tight_layout()
    plt.savefig(OUT / "eda_class_balance.png", dpi=130)
    plt.close()

    return df[TARGET].value_counts().to_dict()


if __name__ == "__main__":
    print("class counts:", run())
