"""
monte_carlo.py
──────────────────────────────────────────────────────────────────────────────
Monte Carlo Stock Price Simulator using Geometric Brownian Motion (GBM)

Theory:
  Under GBM, a stock price S follows:
    dS = μ·S·dt + σ·S·dW
  where:
    μ  = drift (expected annual return)
    σ  = volatility (annual standard deviation of returns)
    dW = Wiener process increment ~ N(0, √dt)

  Discretised for simulation:
    S(t+dt) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z)
  where Z ~ N(0, 1)

  The (μ - σ²/2) term is the Itô correction — it ensures the expected
  value of S grows at rate μ rather than (μ + σ²/2).

Usage:
  python monte_carlo.py                        # default: AAPL, 1000 sims, 252 days
  python monte_carlo.py --ticker TSLA          # different ticker
  python monte_carlo.py --ticker MSFT --sims 5000 --days 504   # 2 years
  python monte_carlo.py --ticker SPY --no-fetch --mu 0.10 --sigma 0.18
  python monte_carlo.py --help
"""

import argparse
import warnings
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from scipy import stats as scipy_stats
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────

DEFAULT_TICKER    = "AAPL"
DEFAULT_SIMS      = 1_000
DEFAULT_DAYS      = 252          # 1 trading year
DEFAULT_HIST_DAYS = 504          # 2 years of history to fit μ, σ
TRADING_DAYS_PA   = 252

# Visual
BG        = "#0a0e1a"
PANEL     = "#111827"
GRID_C    = "#1f2937"
TEXT_C    = "#e2e8f0"
MUTED_C   = "#6b7280"
ACCENT    = "#38bdf8"           # sky blue
BULL_C    = "#34d399"           # emerald
BEAR_C    = "#f87171"           # red
BAND_C    = "#818cf8"           # indigo — confidence band
MEDIAN_C  = "#fbbf24"           # amber — median path
HIST_C    = "#475569"           # slate — historical price


# ─────────────────────────────────────────────────────────────
# DATA & PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────

def fetch_history(ticker: str, days: int) -> pd.Series:
    """Download adjusted closing prices via yfinance."""
    period = f"{max(days // 252 + 1, 2)}y"
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    col = ("Close", ticker) if ("Close", ticker) in df.columns else "Close"
    prices = df[col].dropna()
    return prices.tail(days)


def estimate_parameters(prices: pd.Series) -> tuple[float, float, float]:
    """
    Estimate annualised μ and σ from historical daily log-returns.
    Returns (mu, sigma, S0) where S0 is the last observed price.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu_daily    = log_returns.mean()
    sigma_daily = log_returns.std()
    mu_annual   = mu_daily    * TRADING_DAYS_PA
    sigma_annual= sigma_daily * np.sqrt(TRADING_DAYS_PA)
    S0          = float(prices.iloc[-1])
    return mu_annual, sigma_annual, S0


# ─────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────

def simulate_gbm(S0: float, mu: float, sigma: float,
                 n_days: int, n_sims: int,
                 seed: int = 42) -> np.ndarray:
    """
    Simulate n_sims price paths over n_days using GBM.

    Returns:
      paths : np.ndarray of shape (n_days + 1, n_sims)
              Row 0 is S0 for all paths.
    """
    rng    = np.random.default_rng(seed)
    dt     = 1 / TRADING_DAYS_PA
    drift  = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Random shocks: shape (n_days, n_sims)
    Z      = rng.standard_normal((n_days, n_sims))
    daily_returns = np.exp(drift + diffusion * Z)

    paths  = np.empty((n_days + 1, n_sims))
    paths[0] = S0
    for t in range(1, n_days + 1):
        paths[t] = paths[t - 1] * daily_returns[t - 1]

    return paths


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

def compute_stats(paths: np.ndarray, S0: float,
                  confidence_levels: list = [0.05, 0.25, 0.75, 0.95]
                  ) -> dict:
    """
    Compute terminal price distribution stats and percentile bands.
    """
    terminal      = paths[-1]          # final prices across all sims
    returns       = (terminal - S0) / S0 * 100

    percentiles = {}
    for cl in confidence_levels:
        percentiles[cl] = np.percentile(paths, cl * 100, axis=1)

    return {
        "S0":              S0,
        "n_sims":          paths.shape[1],
        "n_days":          paths.shape[0] - 1,
        "terminal":        terminal,
        "returns":         returns,
        "mean_price":      terminal.mean(),
        "median_price":    np.median(terminal),
        "std_price":       terminal.std(),
        "pct_5":           np.percentile(terminal, 5),
        "pct_25":          np.percentile(terminal, 25),
        "pct_75":          np.percentile(terminal, 75),
        "pct_95":          np.percentile(terminal, 95),
        "prob_profit":     (terminal > S0).mean() * 100,
        "prob_gain_20":    (terminal > S0 * 1.20).mean() * 100,
        "prob_loss_20":    (terminal < S0 * 0.80).mean() * 100,
        "percentile_bands": percentiles,
        "median_path":     np.percentile(paths, 50, axis=1),
    }


# ─────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────

def print_report(ticker: str, mu: float, sigma: float,
                 n_days: int, stats: dict):
    W = 62
    years = n_days / TRADING_DAYS_PA

    print()
    print("╔" + "═" * (W - 2) + "╗")
    print(f"║{'Monte Carlo Price Simulator':^{W-2}}║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  Ticker     : {ticker:<{W-18}}║")
    print(f"║  Horizon    : {n_days} trading days ({years:.1f} yr){'':<{W-43}}║")
    print(f"║  Simulations: {stats['n_sims']:,}{'':<{W-22}}║")
    print(f"║  Ann. Drift : {mu*100:.2f}%{'':<{W-21}}║")
    print(f"║  Ann. Vol   : {sigma*100:.2f}%{'':<{W-21}}║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  Starting price  : ${stats['S0']:.2f}{'':<{W-26}}║")
    print("║" + "─" * (W - 2) + "║")
    print(f"║  Mean   terminal : ${stats['mean_price']:.2f}{'':<{W-26}}║")
    print(f"║  Median terminal : ${stats['median_price']:.2f}{'':<{W-26}}║")
    print(f"║  Std Dev         : ${stats['std_price']:.2f}{'':<{W-26}}║")
    print("║" + "─" * (W - 2) + "║")
    print(f"║  5th  percentile : ${stats['pct_5']:.2f}{'':<{W-26}}║")
    print(f"║  25th percentile : ${stats['pct_25']:.2f}{'':<{W-26}}║")
    print(f"║  75th percentile : ${stats['pct_75']:.2f}{'':<{W-26}}║")
    print(f"║  95th percentile : ${stats['pct_95']:.2f}{'':<{W-26}}║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  P(profit)       : {stats['prob_profit']:.1f}%{'':<{W-25}}║")
    print(f"║  P(gain > 20%)   : {stats['prob_gain_20']:.1f}%{'':<{W-25}}║")
    print(f"║  P(loss > 20%)   : {stats['prob_loss_20']:.1f}%{'':<{W-25}}║")
    print("╚" + "═" * (W - 2) + "╝")
    print()


# ─────────────────────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────────────────────

def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.tick_params(colors=MUTED_C, labelsize=8)
    ax.xaxis.label.set_color(MUTED_C)
    ax.yaxis.label.set_color(MUTED_C)
    if title:
        ax.set_title(title, color=TEXT_C, fontsize=9, pad=7, fontweight="normal")


def plot_simulation(paths: np.ndarray, stats: dict,
                    ticker: str, mu: float, sigma: float,
                    hist_prices: pd.Series | None,
                    output_path: str = "monte_carlo.png",
                    max_plot_paths: int = 300):
    """
    4-panel publication-quality chart:
      Panel A (large, left)  : Fan of simulation paths with confidence bands
      Panel B (top-right)    : Terminal price distribution (histogram + KDE)
      Panel C (mid-right)    : Return distribution with probability zones
      Panel D (bot-right)    : Key probability stats — annotated bar chart
    """
    n_days   = stats["n_days"]
    terminal = stats["terminal"]
    xs       = np.arange(n_days + 1)

    fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            width_ratios=[1.55, 1],
                            height_ratios=[1.1, 1, 1],
                            hspace=0.52, wspace=0.32)

    # ── Panel A: Path fan ──────────────────────────────────────────
    ax_paths = fig.add_subplot(gs[:, 0])
    _style(ax_paths)

    # Draw thin individual paths (sampled subset)
    rng_plot  = np.random.default_rng(0)
    idx_plot  = rng_plot.choice(paths.shape[1],
                                size=min(max_plot_paths, paths.shape[1]),
                                replace=False)
    for i in idx_plot:
        final = paths[-1, i]
        color = BULL_C if final >= stats["S0"] else BEAR_C
        ax_paths.plot(xs, paths[:, i],
                      color=color, alpha=0.04, linewidth=0.5)

    # Confidence bands (filled between percentiles)
    bands = stats["percentile_bands"]
    ax_paths.fill_between(xs, bands[0.05], bands[0.95],
                          color=BAND_C, alpha=0.10, label="5–95th pct")
    ax_paths.fill_between(xs, bands[0.25], bands[0.75],
                          color=BAND_C, alpha=0.20, label="25–75th pct")

    # Percentile lines
    for cl, ls, lw in [(0.05, "--", 0.8), (0.95, "--", 0.8),
                       (0.25, "-",  0.6), (0.75, "-",  0.6)]:
        ax_paths.plot(xs, bands[cl],
                      color=BAND_C, linewidth=lw, linestyle=ls, alpha=0.6)

    # Median path
    ax_paths.plot(xs, stats["median_path"],
                  color=MEDIAN_C, linewidth=1.8,
                  linestyle="-", label="Median path", zorder=5)

    # Historical price context (grey, to the left)
    if hist_prices is not None and len(hist_prices) > 1:
        hist_norm = hist_prices.values
        hist_xs   = np.linspace(-len(hist_norm), 0, len(hist_norm))
        ax_paths.plot(hist_xs, hist_norm,
                      color=HIST_C, linewidth=1.2, alpha=0.7,
                      label="Historical price")
        ax_paths.axvline(0, color=MUTED_C, linewidth=0.8,
                         linestyle=":", alpha=0.6, label="Today")

    # Probability shading — shade the area above/below S0 at end
    ax_paths.axhline(stats["S0"], color=MUTED_C, linewidth=0.8,
                     linestyle=":", alpha=0.5)

    # Annotations
    ax_paths.text(n_days * 0.98, float(stats["median_path"][-1]) * 1.012,
                  f"Median\n${stats['median_path'][-1]:.0f}",
                  color=MEDIAN_C, fontsize=7.5, ha="right",
                  va="bottom", fontweight="bold")

    ax_paths.text(n_days * 0.98, float(bands[0.95][-1]) * 1.012,
                  f"95th pct\n${bands[0.95][-1]:.0f}",
                  color=BAND_C, fontsize=6.5, ha="right", va="bottom", alpha=0.8)

    ax_paths.text(n_days * 0.98, float(bands[0.05][-1]) * 0.985,
                  f"5th pct\n${bands[0.05][-1]:.0f}",
                  color=BAND_C, fontsize=6.5, ha="right", va="top", alpha=0.8)

    ax_paths.set_xlabel("Trading Days", fontsize=9)
    ax_paths.set_ylabel("Price ($)", fontsize=9)
    ax_paths.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    # Legend
    handles, labels = ax_paths.get_legend_handles_labels()
    ax_paths.legend(handles, labels, fontsize=7.5,
                    facecolor=PANEL, edgecolor=GRID_C,
                    labelcolor=TEXT_C, loc="upper left",
                    framealpha=0.9)

    n_years = n_days / TRADING_DAYS_PA
    ax_paths.set_title(
        f"{ticker}  ·  {stats['n_sims']:,} Monte Carlo paths  ·  "
        f"{n_days}d ({n_years:.1f}yr)  ·  "
        f"μ={mu*100:.1f}%  σ={sigma*100:.1f}%",
        color=TEXT_C, fontsize=10, pad=10)

    # ── Panel B: Terminal price histogram + KDE ────────────────────
    ax_dist = fig.add_subplot(gs[0, 1])
    _style(ax_dist, f"Terminal Price Distribution  (day {n_days})")

    n_bins = min(80, max(30, stats["n_sims"] // 30))
    ax_dist.hist(terminal, bins=n_bins, density=True,
                 color=ACCENT, alpha=0.5, edgecolor="none")

    # KDE overlay
    kde_x = np.linspace(terminal.min(), terminal.max(), 500)
    kde   = scipy_stats.gaussian_kde(terminal)
    ax_dist.plot(kde_x, kde(kde_x), color=ACCENT, linewidth=1.8)

    # Shade below 5th / above 95th
    x_bear = kde_x[kde_x <= stats["pct_5"]]
    x_bull = kde_x[kde_x >= stats["pct_95"]]
    ax_dist.fill_between(x_bear, kde(x_bear), color=BEAR_C, alpha=0.35)
    ax_dist.fill_between(x_bull, kde(x_bull), color=BULL_C, alpha=0.35)

    # S0 line
    ax_dist.axvline(stats["S0"],     color=MUTED_C, lw=1.0,
                    ls=":", label=f"S₀ ${stats['S0']:.0f}")
    ax_dist.axvline(stats["median_price"], color=MEDIAN_C, lw=1.2,
                    ls="--", label=f"Median ${stats['median_price']:.0f}")
    ax_dist.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    ax_dist.set_xlabel("Terminal Price", fontsize=8)
    ax_dist.set_ylabel("Density", fontsize=8)
    ax_dist.legend(fontsize=7, facecolor=PANEL,
                   edgecolor=GRID_C, labelcolor=TEXT_C)

    # ── Panel C: Return distribution ──────────────────────────────
    ax_ret = fig.add_subplot(gs[1, 1])
    _style(ax_ret, "Simulated Return Distribution")

    returns = stats["returns"]
    ax_ret.hist(returns, bins=n_bins, density=True,
                color=ACCENT, alpha=0.45, edgecolor="none")

    kde_r_x = np.linspace(returns.min(), returns.max(), 500)
    kde_r   = scipy_stats.gaussian_kde(returns)
    ax_ret.plot(kde_r_x, kde_r(kde_r_x), color=ACCENT, linewidth=1.8)

    # Zone shading
    x_loss = kde_r_x[kde_r_x <= -20]
    x_gain = kde_r_x[kde_r_x >= 20]
    x_neg  = kde_r_x[(kde_r_x > -20) & (kde_r_x < 0)]
    x_pos  = kde_r_x[(kde_r_x > 0)   & (kde_r_x < 20)]
    ax_ret.fill_between(x_loss, kde_r(x_loss), color=BEAR_C, alpha=0.5)
    ax_ret.fill_between(x_gain, kde_r(x_gain), color=BULL_C, alpha=0.5)
    ax_ret.fill_between(x_neg,  kde_r(x_neg),  color=BEAR_C, alpha=0.15)
    ax_ret.fill_between(x_pos,  kde_r(x_pos),  color=BULL_C, alpha=0.15)

    ax_ret.axvline(0,   color=MUTED_C,  lw=0.8, ls=":")
    ax_ret.axvline(-20, color=BEAR_C,   lw=0.8, ls="--", alpha=0.6)
    ax_ret.axvline( 20, color=BULL_C,   lw=0.8, ls="--", alpha=0.6)
    ax_ret.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}%"))
    ax_ret.set_xlabel("Return (%)", fontsize=8)
    ax_ret.set_ylabel("Density",    fontsize=8)

    # Labels for zones
    for x_lbl, label, color in [
        (returns.min() * 0.55, f"Loss>20%\n{stats['prob_loss_20']:.1f}%", BEAR_C),
        (returns.max() * 0.55, f"Gain>20%\n{stats['prob_gain_20']:.1f}%", BULL_C),
    ]:
        ax_ret.text(x_lbl, ax_ret.get_ylim()[1] * 0.6 if ax_ret.get_ylim()[1] else 0.001,
                    label, color=color, fontsize=7,
                    ha="center", va="center", fontweight="bold")

    # ── Panel D: Probability summary ──────────────────────────────
    ax_prob = fig.add_subplot(gs[2, 1])
    _style(ax_prob, "Probability Summary")

    labels  = ["P(profit)", "P(gain\n>20%)", "P(loss\n>20%)", "P(gain\n>50%)"]
    prob_gain_50 = (terminal > stats["S0"] * 1.50).mean() * 100
    values  = [stats["prob_profit"], stats["prob_gain_20"],
               stats["prob_loss_20"], prob_gain_50]
    colors  = [BULL_C, BULL_C, BEAR_C, ACCENT]
    alphas  = [0.85, 0.75, 0.75, 0.75]

    x_pos   = np.arange(len(labels))
    bars    = ax_prob.bar(x_pos, values, color=colors, alpha=0.8,
                          width=0.55, edgecolor=BG, linewidth=0.5)
    ax_prob.axhline(50, color=MUTED_C, lw=0.8, ls=":",
                    alpha=0.7, label="50% baseline")
    ax_prob.set_xticks(x_pos)
    ax_prob.set_xticklabels(labels, fontsize=7.5)
    ax_prob.set_ylim(0, 108)
    ax_prob.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_prob.set_ylabel("Probability", fontsize=8)
    ax_prob.legend(fontsize=7, facecolor=PANEL,
                   edgecolor=GRID_C, labelcolor=TEXT_C)

    for bar, val in zip(bars, values):
        ax_prob.text(bar.get_x() + bar.get_width() / 2,
                     val + 1.5, f"{val:.1f}%",
                     ha="center", va="bottom",
                     fontsize=8, color=TEXT_C, fontweight="bold")

    # ── Supertitle ─────────────────────────────────────────────────
    fig.suptitle(
        f"Monte Carlo Stock Price Simulation  ·  {ticker}  ·  "
        f"GBM with Itô correction",
        color=TEXT_C, fontsize=12, y=1.01, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  Chart saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export_paths_csv(paths: np.ndarray, stats: dict,
                     path: str = "simulation_paths.csv",
                     max_paths: int = 100):
    """
    Export a sample of simulation paths + percentile bands to CSV.
    Keeps the file manageable — full 1000×252 matrix would be huge.
    """
    n_days = stats["n_days"]
    df     = pd.DataFrame(index=range(n_days + 1))
    df.index.name = "day"

    # Percentile bands
    bands = stats["percentile_bands"]
    df["p05"]    = bands[0.05]
    df["p25"]    = bands[0.25]
    df["median"] = stats["median_path"]
    df["p75"]    = bands[0.75]
    df["p95"]    = bands[0.95]

    # Sample paths
    n_export = min(max_paths, paths.shape[1])
    for i in range(n_export):
        df[f"sim_{i+1:04d}"] = paths[:, i]

    df.to_csv(path)
    print(f"  CSV  saved  → {path}  "
          f"({n_export} sample paths + percentile bands)")


def export_terminal_csv(stats: dict, path: str = "terminal_prices.csv"):
    """Export the full terminal price distribution to CSV."""
    df = pd.DataFrame({
        "terminal_price": stats["terminal"],
        "return_pct":     stats["returns"],
    })
    df.to_csv(path, index=False)
    print(f"  CSV  saved  → {path}  "
          f"({len(df):,} terminal prices)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Stock Price Simulator using Geometric Brownian Motion"
    )
    parser.add_argument("--ticker",    type=str,   default=DEFAULT_TICKER,
                        help=f"Stock ticker (default: {DEFAULT_TICKER})")
    parser.add_argument("--sims",      type=int,   default=DEFAULT_SIMS,
                        help=f"Number of simulations (default: {DEFAULT_SIMS})")
    parser.add_argument("--days",      type=int,   default=DEFAULT_DAYS,
                        help=f"Forecast horizon in trading days (default: {DEFAULT_DAYS})")
    parser.add_argument("--hist",      type=int,   default=DEFAULT_HIST_DAYS,
                        help="Days of history to fit μ and σ (default: 504)")
    parser.add_argument("--mu",        type=float, default=None,
                        help="Override annual drift (e.g. 0.10 for 10%%)")
    parser.add_argument("--sigma",     type=float, default=None,
                        help="Override annual volatility (e.g. 0.20 for 20%%)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-fetch",  action="store_true",
                        help="Skip yfinance fetch — requires --mu and --sigma")
    parser.add_argument("--no-chart",  action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--no-csv",    action="store_true",
                        help="Skip CSV export")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║      Monte Carlo Stock Price Simulator (GBM)        ║")
    print("╚══════════════════════════════════════════════════════╝")

    hist_prices = None

    if args.no_fetch:
        if args.mu is None or args.sigma is None:
            print("  Error: --no-fetch requires --mu and --sigma.")
            sys.exit(1)
        mu    = args.mu
        sigma = args.sigma
        S0    = 100.0
        print(f"  Using manual parameters: μ={mu:.2%}  σ={sigma:.2%}  S0=${S0:.2f}")
    else:
        print(f"  Fetching {args.ticker} history ...", end=" ", flush=True)
        try:
            hist_prices = fetch_history(args.ticker, args.hist)
        except Exception as e:
            print(f"\n  Error: {e}")
            sys.exit(1)
        mu, sigma, S0 = estimate_parameters(hist_prices)
        print(f"{len(hist_prices)} days fetched ✓")
        print(f"  Estimated: μ={mu:.2%}  σ={sigma:.2%}  S0=${S0:.2f}")

    # Override if provided
    if args.mu    is not None: mu    = args.mu
    if args.sigma is not None: sigma = args.sigma

    print(f"  Running {args.sims:,} simulations × "
          f"{args.days} days ...", end=" ", flush=True)

    paths = simulate_gbm(S0, mu, sigma, args.days, args.sims, seed=args.seed)
    print("done ✓")

    st = compute_stats(paths, S0)
    print_report(args.ticker, mu, sigma, args.days, st)

    safe = args.ticker.replace("=", "").replace("/", "").replace("^", "")
    if not args.no_chart:
        plot_simulation(
            paths, st, args.ticker, mu, sigma, hist_prices,
            output_path=f"monte_carlo_{safe}.png"
        )
    if not args.no_csv:
        export_paths_csv(paths, st, path=f"paths_{safe}.csv")
        export_terminal_csv(st,      path=f"terminal_{safe}.csv")

    print("  Done.\n")


if __name__ == "__main__":
    main()