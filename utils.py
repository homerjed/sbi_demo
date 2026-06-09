import numpy as np
import pandas as pd


def make_df(samples, log_probs, parameter_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(samples, columns=parameter_names)
    df = df.assign(log_posterior=log_probs)
    return df


def marker(p, parameter_names):
    return dict(zip(parameter_names, np.asarray(p)))


def center_triangle_plot(fig, center, covariance, *, n_sigma: float = 4.0):
    center = np.asarray(center)
    covariance = np.asarray(covariance)
    n_dim = center.size
    widths = n_sigma * np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    if len(fig.axes) < n_dim * n_dim:
        return

    for row in range(n_dim):
        for col in range(n_dim):
            ax = fig.axes[row * n_dim + col]
            if row < col:
                continue

            x_width = widths[col]
            if np.isfinite(x_width) and x_width > 0:
                ax.set_xlim(center[col] - x_width, center[col] + x_width)

            if row > col:
                y_width = widths[row]
                if np.isfinite(y_width) and y_width > 0:
                    ax.set_ylim(center[row] - y_width, center[row] + y_width)


def save_npz(path, **items):
    arrays = {}
    for key, value in items.items():
        if value is None:
            continue
        arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)
