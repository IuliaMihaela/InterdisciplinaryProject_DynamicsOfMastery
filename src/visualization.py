# plot_learning_curve()
# plot_model_fit()
# plot_residuals()
import matplotlib.pyplot as plt


def plot_learning_curve(x, y, player_id=None):
    """
    Plot raw learning trajectory.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        y,
        marker=".",
        linestyle="-",
        alpha=0.7
    )

    ax.set_xlabel("Solve Index")
    ax.set_ylabel("Best Solve Time (seconds)")
    ax.set_title(
        f"Learning Trajectory ({player_id})"
        if player_id
        else "Learning Trajectory"
    )

    return fig, ax



def plot_model_fits(
    x,
    y,
    power_pred,
    exp_pred,
    player_id=None
):
    """
    Plot observed data together with fitted models.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        x,
        y,
        s=10,
        alpha=0.4,
        label="Observed"
    )

    ax.plot(
        x,
        power_pred,
        linewidth=2,
        label="Power Law"
    )

    ax.plot(
        x,
        exp_pred,
        linewidth=2,
        label="Exponential"
    )

    ax.set_xlabel("Solve Index")
    ax.set_ylabel("Best Solve Time (seconds)")

    title = "Learning Curve Model Comparison"
    if player_id:
        title += f" ({player_id})"

    ax.set_title(title)

    ax.legend()

    return fig, ax

def plot_residuals(
    x,
    residuals,
    player_id=None
):
    """
    Plot residual series.
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, residuals)

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_xlabel("Solve Index")
    ax.set_ylabel("Residual")

    title = "Residuals"
    if player_id:
        title += f" ({player_id})"

    ax.set_title(title)

    return fig, ax