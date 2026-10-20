"""Top 1% wealth and income shares in the US and Europe, from WID via Our World in Data."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" Settings """

def settings():
    """Series, countries and colors for the figures."""

    series = {
        'wealth': ('wealth-share-richest-1-percent', 'Top 1% wealth share (%)', 1900),
        'income': ('income-share-top-1-before-tax-wid', 'Top 1% pre-tax income share (%)', 1910),
    }
    countries = ['United States', 'France', 'United Kingdom', 'Germany', 'Sweden', 'Denmark']
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
    return series, countries, colors

""" Data """

def load(slug, folder):
    """Download an OWID grapher series once and return it as a long dataframe."""

    path = os.path.join(folder, f'{slug}.csv')
    if not os.path.exists(path):
        url = f'https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true'
        pd.read_csv(url).to_csv(path, index=False)

    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[-1]: 'share'})
    return df

""" Figure """

def spread_labels(y, min_gap):
    """Push label positions apart so that neighbours are at least min_gap away."""

    order = np.argsort(y)
    y_new = np.array(y, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k-1], order[k]
        y_new[j] = max(y_new[j], y_new[i] + min_gap)
    return y_new

def plot_shares(df, countries, colors, ylabel, first_year, path):
    """Line chart of a top share by country, with direct labels at the line ends."""

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ends = []
    for country, color in zip(countries, colors):

        # a. series from first_year, broken where data are missing for more than 10 years
        s = df[(df.entity == country) & (df.year >= first_year)].set_index('year')['share'].dropna()
        years = s.index.values
        gaps = np.where(np.diff(years) > 10)[0]
        x = np.insert(years.astype(float), gaps + 1, np.nan)
        y = np.insert(s.values, gaps + 1, np.nan)

        # b. plot
        ax.plot(x, y, color=color, lw=2, label=country)
        ends.append((years[-1], s.values[-1], country, color))

    # c. direct labels, spread to avoid overlap
    y_label = spread_labels([e[1] for e in ends], min_gap=0.08*ax.get_ylim()[1])
    for (year, _, country, color), yl in zip(ends, y_label):
        ax.text(year + 1.5, yl, country, color='#3a3a38', va='center', fontsize=9)
        ax.plot([year + 0.3, year + 1.3], [yl, yl], color=color, lw=2)

    ax.set_ylabel(ylabel)
    ax.set_xlim(first_year, 2050)
    ax.set_xticks(range(first_year, 2021, 20))
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#e4e3df', lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.text(0, -0.36, 'Source: World Inequality Database, via Our World in Data.',
            transform=ax.transAxes, fontsize=8, color='#6b6a65')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def main():
    """Download the series and save one figure per series in tex/figs."""

    series, countries, colors = settings()
    folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(folder, 'data')
    os.makedirs(data_folder, exist_ok=True)

    for name, (slug, ylabel, first_year) in series.items():
        df = load(slug, data_folder)
        path = os.path.join(folder, 'tex', 'figs', f'top1_{name}_share.pdf')
        plot_shares(df, countries, colors, ylabel, first_year, path)

if __name__ == '__main__':
    main()
