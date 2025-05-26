import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_comparison(
    df,
    x: str,
    y: str,
    hue: str,
    title: str = "Metric Comparison Across Categories",
    y_label: str = None,
    x_label: str = None,
    dot_size: int = 50,
    figsize: tuple = (12, 6),
    rotate_x: bool = True
):
    """
    Plots a scatter plot to compare metrics across categories.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to plot.
    - x (str): Column name to plot on the x-axis (typically a category like country or segment).
    - y (str): Column name to plot on the y-axis (the metric being compared).
    - hue (str): Column used to color-code the points (e.g., method or metric type).
    - title (str): Plot title.
    - y_label (str): Y-axis label. Defaults to y column name.
    - x_label (str): X-axis label. Defaults to x column name.
    - dot_size (int): Size of the dots.
    - figsize (tuple): Width and height of the figure.
    - rotate_x (bool): Whether to rotate x-axis labels for readability.

    Returns:
    - None
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=hue, marker='o', s=dot_size)

    plt.title(title, fontsize=14)
    plt.ylabel(y_label if y_label else y.capitalize())
    plt.xlabel(x_label if x_label else x.capitalize())

    if rotate_x:
        plt.xticks(rotation=90)

    plt.legend(title=hue.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()




def plot_metric_trends_by_country(
    df,
    x="year",
    y="value",
    hue="method",
    country_col="country",
    title="Metric Trends by Country",
    y_label="Value",
    figsize=(10, 3.5),
    dot_size=80,
    linewidth=1.5,
    legend_fontsize=9
):
    sns.set(style="whitegrid", palette="colorblind")

    countries = df[country_col].dropna().unique()
    n_countries = len(countries)

    fig, axes = plt.subplots(
        nrows=n_countries,
        ncols=1,
        figsize=(figsize[0], figsize[1] * n_countries),
        sharex=True
    )
    if n_countries == 1:
        axes = [axes]

    # 1. Hidden axis to generate legend handles
    fig_temp, ax_temp = plt.subplots()
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        ax=ax_temp,
        legend=True
    )
    handles, labels = ax_temp.get_legend_handles_labels()
    plt.close(fig_temp)  # Don't display temp plot

    # 2. Draw each country subplot without legend
    for ax, country in zip(axes, countries):
        df_country = df[df[country_col] == country]
        sns.lineplot(
            data=df_country,
            x=x,
            y=y,
            hue=hue,
            marker='o',
            linewidth=linewidth,
            markersize=dot_size**0.5,
            ax=ax,
            legend=False
        )
        ax.set_title(country, fontsize=12, pad=6)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    axes[-1].set_xlabel(x.capitalize())

    fig.suptitle(title, fontsize=16, y=1.03)

    fig.legend(
        handles=handles,
        labels=labels,
        title=hue.capitalize(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),  # lower than before
        ncol=len(labels),
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 1,
        frameon=False
    )

    # Reduce top margin to bring first subplot closer
    fig.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.01)

    plt.show()
