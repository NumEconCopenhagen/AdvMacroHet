"""Real 5- and 10-year Treasury yields and CPI inflation in the US since 2003."""

import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

""" Data """

def load_fred(series, folder):
    """Download a FRED series once and return it as a dated series."""

    path = os.path.join(folder, f'{series}.csv')
    if not os.path.exists(path):
        pd.read_csv(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}').to_csv(path, index=False)

    df = pd.read_csv(path, parse_dates=['observation_date']).set_index('observation_date')
    return pd.to_numeric(df[series], errors='coerce').dropna()

""" Figure """

def style_axis(ax, title):
    """Recessive grid and axes, title on top."""

    ax.set_title(title, fontsize=11, loc='left', color='#0b0b0b')
    ax.axhline(0, color='#c3c2b7', lw=0.8)
    ax.grid(axis='y', color='#e1e0d9', lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e', labelsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def plot(yields, inflation, path):
    """Real yields on the left, inflation on the right."""

    fig, (ax_r, ax_pi) = plt.subplots(1, 2, figsize=(6.4, 3.2))

    # a. real yields, direct labels at the end of each line
    for (label, s), color, offset in zip(yields.items(), ['#2a78d6', '#eb6834'], [-0.6, 0.6]):
        ax_r.plot(s.index, s, color=color, lw=1.6, label=label)
        ax_r.text(s.index[-1] + pd.DateOffset(months=4), s.iloc[-1] + offset, label, color='#52514e', fontsize=9, va='center')

    style_axis(ax_r, 'Real Treasury yields (%)')
    ax_r.legend(frameon=False, fontsize=9, loc='lower left')
    ax_r.set_xlim(yields.index[0], yields.index[-1] + pd.DateOffset(years=3))

    # b. inflation with the 2% reference
    ax_pi.plot(inflation.index, inflation, color='#52514e', lw=1.6)
    ax_pi.axhline(2, color='#898781', lw=0.8, ls='--')
    ax_pi.text(inflation.index[-1] + pd.DateOffset(months=4), 2.5, '2%', color='#898781', fontsize=9, va='center')
    style_axis(ax_pi, 'CPI inflation, 12 months (%)')
    ax_pi.set_xlim(inflation.index[0], inflation.index[-1] + pd.DateOffset(years=3))

    fig.text(0.01, 0.01, 'Source: FRED (FII5, FII10, CPIAUCSL), monthly.', fontsize=8, color='#898781')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)

def main():
    """Download the data and save the figure in tex/figs."""

    folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(folder, 'data')
    os.makedirs(data_folder, exist_ok=True)

    yields = pd.DataFrame({'5-year': load_fred('FII5', data_folder), '10-year': load_fred('FII10', data_folder)}).dropna()
    inflation = 100*load_fred('CPIAUCSL', data_folder).pct_change(12).loc[yields.index[0]:]

    plot(yields, inflation, os.path.join(folder, 'tex', 'figs', 'real_rates_inflation.pdf'))

if __name__ == '__main__':
    main()
