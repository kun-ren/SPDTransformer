from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_run_histories(history_path: str):
    """
    Given one history.csv inside a fold directory, automatically load
    all fold histories belonging to the same run.

    Expected structure:
        run_xxx/
            fold_01/history.csv
            fold_02/history.csv
            ...
            fold_05/history.csv
    """

    history_path = Path(history_path)

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    # Example:
    # history_path = run_xxx/fold_02/history.csv
    # fold_dir     = run_xxx/fold_02
    # run_dir      = run_xxx
    fold_dir = history_path.parent
    run_dir = fold_dir.parent

    fold_dirs = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("fold_")
    )

    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under: {run_dir}")

    histories = {}

    for fold in fold_dirs:
        csv_path = fold / "history.csv"

        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist. Skipping.")
            continue

        df = pd.read_csv(csv_path)

        # If history.csv does not explicitly contain epoch,
        # generate epoch numbers starting from 1.
        if "epoch" not in df.columns:
            df.insert(0, "epoch", np.arange(1, len(df) + 1))

        histories[fold.name] = df

    if not histories:
        raise RuntimeError(f"No history.csv files found under: {run_dir}")

    print(f"Run directory: {run_dir}")
    print(f"Loaded {len(histories)} folds:")
    for fold_name, df in histories.items():
        print(
            f"  {fold_name}: "
            f"{len(df)} epochs, "
            f"columns={list(df.columns)}"
        )

    return run_dir, histories


def find_column(df: pd.DataFrame, candidates):
    """
    Find the first existing column from a list of possible names.
    """

    for col in candidates:
        if col in df.columns:
            return col

    return None


def collect_metric(histories, column_candidates):
    """
    Collect the same metric from all folds.

    Different folds may have different epoch lengths because of
    early stopping. Missing epochs are represented by NaN.
    """

    fold_series = {}

    for fold_name, df in histories.items():

        column = find_column(df, column_candidates)

        if column is None:
            continue

        values = df[column].to_numpy(dtype=float)

        fold_series[fold_name] = values

    if not fold_series:
        return None, None, None

    max_epochs = max(len(v) for v in fold_series.values())

    matrix = np.full(
        (len(fold_series), max_epochs),
        np.nan,
        dtype=float,
    )

    for i, values in enumerate(fold_series.values()):
        matrix[i, :len(values)] = values

    epochs = np.arange(1, max_epochs + 1)

    # Mean over available folds
    mean = np.nanmean(matrix, axis=0)

    # Number of available folds for every epoch
    n = np.sum(~np.isnan(matrix), axis=0)

    # Sample standard deviation (ddof=1)
    std = np.full(max_epochs, np.nan)

    valid = n >= 2

    std[valid] = np.nanstd(
        matrix[:, valid],
        axis=0,
        ddof=1,
    )

    return epochs, matrix, (mean, std, n)


def plot_metric(
    ax,
    histories,
    train_candidates,
    val_candidates,
    title,
    ylabel,
    percentage=False,
):
    """
    Plot individual folds + mean ± std for train and validation metrics.
    """

    train_result = collect_metric(histories, train_candidates)
    val_result = collect_metric(histories, val_candidates)

    train_epochs, train_matrix, train_stats = train_result
    val_epochs, val_matrix, val_stats = val_result

    if train_matrix is None and val_matrix is None:
        ax.set_visible(False)
        return

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    if train_matrix is not None:

        train_mean, train_std, train_n = train_stats

        if percentage:
            train_matrix = train_matrix * 100
            train_mean = train_mean * 100
            train_std = train_std * 100

        # Individual folds
        for fold_values in train_matrix:
            ax.plot(
                train_epochs,
                fold_values,
                alpha=0.18,
                linewidth=0.8,
            )

        # Mean
        ax.plot(
            train_epochs,
            train_mean,
            linewidth=2.0,
            label="Train mean",
        )

        # Mean ± SD
        valid_std = ~np.isnan(train_std)

        ax.fill_between(
            train_epochs[valid_std],
            (train_mean - train_std)[valid_std],
            (train_mean + train_std)[valid_std],
            alpha=0.15,
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    if val_matrix is not None:

        val_mean, val_std, val_n = val_stats

        if percentage:
            val_matrix = val_matrix * 100
            val_mean = val_mean * 100
            val_std = val_std * 100

        for fold_values in val_matrix:
            ax.plot(
                val_epochs,
                fold_values,
                alpha=0.18,
                linewidth=0.8,
                linestyle="--",
            )

        ax.plot(
            val_epochs,
            val_mean,
            linewidth=2.0,
            linestyle="--",
            label="Validation mean",
        )

        valid_std = ~np.isnan(val_std)

        ax.fill_between(
            val_epochs[valid_std],
            (val_mean - val_std)[valid_std],
            (val_mean + val_std)[valid_std],
            alpha=0.15,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    ax.grid(
        True,
        alpha=0.25,
        linestyle="--",
        linewidth=0.5,
    )

    ax.legend(frameon=False)


def plot_training_history(
    history_path: str,
    output_name="training_curves.png",
    dpi=300,
):
    """
    Load all folds of a run and generate publication-ready
    training curves.

    Output:
        run_xxx/training_curves.png
    """

    run_dir, histories = load_run_histories(history_path)

    # ---------------------------------------------------------
    # Print detected columns
    # ---------------------------------------------------------
    first_df = next(iter(histories.values()))

    print("\nAvailable history columns:")
    for col in first_df.columns:
        print(f"  {col}")

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------
    plot_metric(
        axes[0],
        histories,
        train_candidates=[
            "train_loss",
            "loss",
        ],
        val_candidates=[
            "val_loss",
            "valid_loss",
            "test_loss",
        ],
        title="Loss",
        ylabel="Loss",
        percentage=False,
    )

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------
    plot_metric(
        axes[1],
        histories,
        train_candidates=[
            "train_acc",
            "train_accuracy",
            "accuracy",
        ],
        val_candidates=[
            "val_acc",
            "val_accuracy",
            "valid_acc",
            "test_acc",
            "test_accuracy",
        ],
        title="Accuracy",
        ylabel="Accuracy (%)",
        percentage=True,
    )

    # ---------------------------------------------------------
    # Macro-F1
    # ---------------------------------------------------------
    plot_metric(
        axes[2],
        histories,
        train_candidates=[
            "train_macro_f1",
            "train_mf1",
            "macro_f1",
        ],
        val_candidates=[
            "val_macro_f1",
            "val_mf1",
            "valid_macro_f1",
            "test_macro_f1",
            "test_mf1",
        ],
        title="Macro-F1",
        ylabel="Macro-F1 (%)",
        percentage=True,
    )

    fig.tight_layout()

    output_path = run_dir / output_name

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    # Also save vector PDF for thesis
    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved PNG: {output_path}")
    print(f"Saved PDF: {pdf_path}")

    return output_path, pdf_path


if __name__ == "__main__":

    history_path = (
        r"D:\thesis\files\run_008_5b7b8e94f4"
        r"\fold_02\history.csv"
    )

    plot_training_history(
        history_path=history_path,
        output_name="training_curves.png",
        dpi=300,
    )