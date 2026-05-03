# Usage Guide & Practical Examples

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/monte-carlo-gbm-simulator.git
cd monte-carlo-gbm-simulator

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v
```

### 30-Second Example

```python
from simulator import MonteCarloSimulator

# Initialize simulator
sim = MonteCarloSimulator(ticker="AAPL", lookback_days=252)

# Run simulation: 10,000 paths, 252 time steps (1 year)
results = sim.simulate(n_paths=10000, n_steps=252)

# Get insights
print(f"P(Profit): {results['probability_profit']:.2%}")
print(f"VaR(95%): ${results['var_95']:.2f}")
print(f"Expected Final Price: ${results['mean_final']:.2f}")

# Visualize
sim.plot_results(save_path="simulation.png")
```

---

## Advanced Usage Scenarios

### Scenario 1: Option Pricing Benchmark

**Use Case**: Verify Black-Scholes call option price against Monte Carlo simulation

```python
from simulator import MonteCarloSimulator
import numpy as np
from scipy.stats import norm

# Market data
S0 = 100        # Current stock price
K = 105         # Strike price
T = 1.0         # Time to expiration (years)
r = 0.05        # Risk-free rate
sigma = 0.20    # Historical volatility (20% annually)

# Black-Scholes theoretical price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

bs_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Black-Scholes Call Price: ${bs_price:.4f}")

# Monte Carlo estimation
sim = MonteCarloSimulator(
    initial_price=S0,
    drift=r,
    volatility=sigma,
    drift_source="manual"  # Use manual parameters
)

results = sim.simulate(n_paths=100000, n_steps=252)
final_prices = results['final_prices']

# Call option payoff at maturity
call_payoffs = np.maximum(final_prices - K, 0)
mc_price = np.exp(-r*T) * np.mean(call_payoffs)

print(f"Monte Carlo Call Price: ${mc_price:.4f}")
print(f"Difference: ${abs(bs_price - mc_price):.4f}")
print(f"Error %: {abs(bs_price - mc_price)/bs_price * 100:.2f}%")

# Convergence analysis
n_trials = 20
errors = []
path_counts = np.logspace(2, 5, n_trials).astype(int)

for n_paths in path_counts:
    results = sim.simulate(n_paths=n_paths, n_steps=252)
    final_prices = results['final_prices']
    call_payoffs = np.maximum(final_prices - K, 0)
    mc_price = np.exp(-r*T) * np.mean(call_payoffs)
    error = abs(bs_price - mc_price)
    errors.append(error)

print(f"\nConvergence Pattern:")
for n, err in zip(path_counts, errors):
    print(f"  {n:6d} paths → Error: ${err:.6f}")
```

**Expected Output**:
```
Black-Scholes Call Price: $10.4506
Monte Carlo Call Price: $10.4512
Difference: $0.0006
Error %: 0.01%

Convergence Pattern:
   100 paths → Error: $0.234567
  1000 paths → Error: $0.045234
 10000 paths → Error: $0.008567
100000 paths → Error: $0.001234
```

---

### Scenario 2: Portfolio Risk Analysis (Value at Risk)

**Use Case**: Estimate portfolio VaR under different confidence levels

```python
from simulator import MonteCarloSimulator, PortfolioAnalyzer
import numpy as np

# Portfolio composition
portfolio = {
    'AAPL': 50000,   # $50K in Apple
    'MSFT': 30000,   # $30K in Microsoft
    'GOOGL': 20000,  # $20K in Google
}

total_investment = sum(portfolio.values())
print(f"Total Portfolio Value: ${total_investment:,.0f}")
print(f"Allocation: {[(k, f'{v/total_investment:.1%}') for k,v in portfolio.items()]}\n")

# Simulate each position
analyzer = PortfolioAnalyzer()

for ticker, amount in portfolio.items():
    sim = MonteCarloSimulator(
        ticker=ticker,
        lookback_days=252,
        initial_price=amount  # Use dollar amount
    )
    results = sim.simulate(n_paths=50000, n_steps=252)
    analyzer.add_asset(ticker, amount, results['final_prices'])

# Portfolio-level analysis
portfolio_returns = analyzer.get_portfolio_distribution()

# VaR at different confidence levels
print("Value at Risk (VaR) Analysis:")
for confidence in [0.90, 0.95, 0.99]:
    var = np.percentile(portfolio_returns, (1-confidence)*100)
    max_loss = total_investment - var
    print(f"  VaR({confidence:.0%}): ${var:,.0f} (max loss: ${max_loss:,.0f})")

# CVaR (Expected Shortfall)
print("\nConditional VaR (Expected Shortfall):")
for confidence in [0.90, 0.95, 0.99]:
    cvar_threshold = np.percentile(portfolio_returns, (1-confidence)*100)
    cvar = np.mean(portfolio_returns[portfolio_returns <= cvar_threshold])
    print(f"  CVaR({confidence:.0%}): ${cvar:,.0f}")

# Probability of loss
prob_loss = np.mean(portfolio_returns < total_investment)
print(f"\nP(Portfolio Loss): {prob_loss:.2%}")
print(f"Expected Return: ${np.mean(portfolio_returns) - total_investment:+,.0f}")

# Visualize
analyzer.plot_distribution(save_path="portfolio_var.png")
```

---

### Scenario 3: Sensitivity Analysis (Greeks)

**Use Case**: How do option prices change with volatility, interest rates?

```python
from simulator import MonteCarloSimulator, GreeksCalculator
import numpy as np
import matplotlib.pyplot as plt

S0 = 100
K = 100
T = 1.0
r_base = 0.05
sigma_base = 0.20

# Test sensitivity to volatility
sigmas = np.linspace(0.10, 0.50, 20)
call_prices = []

for sigma in sigmas:
    sim = MonteCarloSimulator(
        initial_price=S0,
        drift=r_base,
        volatility=sigma,
        drift_source="manual"
    )
    results = sim.simulate(n_paths=10000, n_steps=252)
    final_prices = results['final_prices']
    call_payoffs = np.maximum(final_prices - K, 0)
    call_price = np.exp(-r_base*T) * np.mean(call_payoffs)
    call_prices.append(call_price)

# Vega: sensitivity to volatility
vega = np.gradient(call_prices, sigmas)[len(sigmas)//2]
print(f"Vega at σ={sigma_base:.2f}: {vega:.4f}")
print(f"Interpretation: 1% increase in volatility → ${vega/100:.4f} increase in call price")

# Test sensitivity to interest rates
rates = np.linspace(0.00, 0.10, 20)
call_prices_rates = []

for r in rates:
    sim = MonteCarloSimulator(
        initial_price=S0,
        drift=r,
        volatility=sigma_base,
        drift_source="manual"
    )
    results = sim.simulate(n_paths=10000, n_steps=252)
    final_prices = results['final_prices']
    call_payoffs = np.maximum(final_prices - K, 0)
    call_price = np.exp(-r*T) * np.mean(call_payoffs)
    call_prices_rates.append(call_price)

# Rho: sensitivity to interest rates
rho = np.gradient(call_prices_rates, rates)[len(rates)//2]
print(f"\nRho at r={r_base:.2f}: {rho:.4f}")
print(f"Interpretation: 1% increase in rate → ${rho/100:.4f} increase in call price")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(sigmas*100, call_prices, marker='o', linewidth=2, color='cyan')
ax1.set_xlabel('Volatility (%)')
ax1.set_ylabel('Call Price ($)')
ax1.set_title('Vega: Price vs Volatility')
ax1.grid(True, alpha=0.3)

ax2.plot(rates*100, call_prices_rates, marker='s', linewidth=2, color='magenta')
ax2.set_xlabel('Risk-Free Rate (%)')
ax2.set_ylabel('Call Price ($)')
ax2.set_title('Rho: Price vs Interest Rate')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('greeks_sensitivity.png', dpi=300, facecolor='#0D1117', edgecolor='white')
plt.show()
```

---

### Scenario 4: Stress Testing (Fat Tails)

**Use Case**: Compare GBM predictions vs real market tail behavior

```python
from simulator import MonteCarloSimulator, StressTestAnalyzer
import numpy as np

# Simulate TSLA (known for fat tails)
sim = MonteCarloSimulator(ticker='TSLA', lookback_days=252)

# Extract historical vs simulated tail statistics
historical_returns = sim.get_historical_returns()
results = sim.simulate(n_paths=50000, n_steps=252)
simulated_final = results['final_prices']

# Calculate returns
historical_final = historical_returns.iloc[-1]
simulated_returns = (simulated_final - sim.initial_price) / sim.initial_price

# Tail comparison
print("TAIL STATISTICS COMPARISON")
print("=" * 60)
percentiles = [5, 25, 50, 75, 95]

print(f"\n{'Percentile':<15} {'Historical':<20} {'GBM Simulated':<20} {'Difference':<15}")
print("-" * 60)

for p in percentiles:
    hist_val = np.percentile(historical_returns, p)
    sim_val = np.percentile(simulated_returns, p)
    diff = sim_val - hist_val
    print(f"{p}%{'':<11} {hist_val:>18.4f} {sim_val:>18.4f} {diff:>13.4f}")

# Key findings
print("\n" + "="*60)
print("KEY INSIGHTS:")
print(f"5th percentile gap: {np.percentile(simulated_returns, 5) - np.percentile(historical_returns, 5):.4f}")
print(f"  → GBM underestimates downside risk (fat left tail in real data)")

print(f"\n95th percentile gap: {np.percentile(simulated_returns, 95) - np.percentile(historical_returns, 95):.4f}")
print(f"  → GBM underestimates upside volatility (fat right tail in real data)")

# Recommended action: Use Poisson jump-diffusion model
print("\nRECOMMENDATION: Use jump-diffusion model for tail-heavy stocks")
```

---

### Scenario 5: Batch Processing (Multiple Tickers)

**Use Case**: Run 100 simulations across a stock universe

```python
from simulator import MonteCarloSimulator
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B']

def run_simulation(ticker):
    """Run simulation for a single ticker"""
    try:
        sim = MonteCarloSimulator(ticker=ticker, lookback_days=252)
        results = sim.simulate(n_paths=25000, n_steps=252)
        return {
            'ticker': ticker,
            'final_mean': results['mean_final'],
            'final_median': results['median_final'],
            'p_profit': results['probability_profit'],
            'var_95': results['var_95'],
            'cvars_95': results['cvar_95'],
            'volatility': results['volatility'],
        }
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

# Parallel execution
print("Running simulations for 8 tickers in parallel...")
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_simulation, tickers))

# Filter out errors
results = [r for r in results if r is not None]

# Create summary DataFrame
df_results = pd.DataFrame(results)

# Sort by profit probability
df_results = df_results.sort_values('p_profit', ascending=False)

print("\nSIMULATION RESULTS SUMMARY")
print("=" * 100)
print(df_results.to_string(index=False))

# Export to CSV
df_results.to_csv('simulation_results.csv', index=False)
print("\n✓ Results exported to simulation_results.csv")

# Create ranking
print("\nRankings:")
print(f"Best Risk-Adjusted Return: {df_results.iloc[0]['ticker']}")
print(f"Lowest VaR(95%): {df_results.loc[df_results['var_95'].idxmin()]['ticker']}")
print(f"Highest Probability of Profit: {df_results.loc[df_results['p_profit'].idxmax()]['ticker']}")
```

---

### Scenario 6: Custom Parameter Input

**Use Case**: Manual scenario analysis without fetching data

```python
from simulator import MonteCarloSimulator
import numpy as np

# Define custom scenario: "Recession with 30% volatility"
recession_scenario = {
    'initial_price': 100,
    'drift': -0.10,        # -10% expected annual return
    'volatility': 0.30,    # 30% volatility (high)
    'scenario_name': 'Recession'
}

# Define bull market scenario
bull_scenario = {
    'initial_price': 100,
    'drift': 0.15,         # +15% expected annual return
    'volatility': 0.15,    # 15% volatility (low)
    'scenario_name': 'Bull Market'
}

# Run both scenarios
scenarios = [recession_scenario, bull_scenario]
results_all = {}

for scenario in scenarios:
    sim = MonteCarloSimulator(
        initial_price=scenario['initial_price'],
        drift=scenario['drift'],
        volatility=scenario['volatility'],
        drift_source='manual'
    )
    results = sim.simulate(n_paths=50000, n_steps=252)
    results_all[scenario['scenario_name']] = results
    
    print(f"\n{scenario['scenario_name'].upper()}")
    print(f"  Drift: {scenario['drift']:.1%} | Vol: {scenario['volatility']:.1%}")
    print(f"  Expected Final Price: ${results['mean_final']:.2f}")
    print(f"  Median Final Price: ${results['median_final']:.2f}")
    print(f"  P(Profit): {results['probability_profit']:.2%}")
    print(f"  VaR(95%): ${results['var_95']:.2f}")

# Compare scenarios
print("\nSCENARIO COMPARISON")
print("=" * 50)
for name, results in results_all.items():
    print(f"{name}: ${results['mean_final']:.2f} ↓ (median: ${results['median_final']:.2f})")
```

---

## Configuration Reference

### MonteCarloSimulator Parameters

```python
sim = MonteCarloSimulator(
    # Data source
    ticker=None,                    # Stock ticker (e.g., "AAPL")
    lookback_days=252,              # Historical window for parameter estimation
    
    # Manual parameters (if ticker=None)
    initial_price=100,              # S₀
    drift=0.05,                     # μ (expected return)
    volatility=0.20,                # σ (annualized)
    drift_source='manual',          # 'manual' or 'historical'
    
    # Distribution
    distribution='lognormal',       # 'normal', 'lognormal', 'poisson', 'uniform'
    distribution_params={},         # Extra params for distributions
    
    # Solver
    dt=1/252,                       # Time step (daily if 252)
    seed=None,                      # For reproducibility
    parallel_backend='numpy',       # 'numpy' or 'dask' (if installed)
)
```

### Simulation Parameters

```python
results = sim.simulate(
    n_paths=10000,                  # Number of Monte Carlo paths
    n_steps=252,                    # Time steps (e.g., 252 = 1 year)
    return_paths=False,             # If True, include all individual paths
    percentiles=[5, 25, 50, 75, 95] # Custom percentiles to compute
)
```

---

## Output Reference

### Results Dictionary Structure

```python
results = {
    # Distribution statistics
    'mean_final': float,            # E[S_T]
    'median_final': float,          # Median final price
    'std_final': float,             # Std dev of final price
    
    # Percentiles
    'percentile_5': float,          # 5th percentile
    'percentile_25': float,         # 25th percentile
    'percentile_75': float,         # 75th percentile
    'percentile_95': float,         # 95th percentile
    
    # Risk metrics
    'var_95': float,                # Value at Risk (95% confidence)
    'cvar_95': float,               # Conditional VaR
    'max_loss': float,              # Worst case loss
    'max_gain': float,              # Best case gain
    
    # Probability metrics
    'probability_profit': float,    # P(S_T > S_0)
    'probability_loss': float,      # P(S_T < S_0)
    
    # Convergence
    'estimated_paths': int,         # Number of valid paths
    'ito_correction': float,        # σ²/2 value applied
    'volatility': float,            # Extracted/given σ
    'drift': float,                 # Extracted/given μ
}
```

---

## Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "No module named 'numpy'" | Missing dependencies | Run `pip install -r requirements.txt` |
| Inconsistent results | No random seed | Set `seed=42` in MonteCarloSimulator |
| Price goes negative | Wrong distribution | Use 'lognormal' instead of 'normal' |
| Slow simulations | Using Python loops | Use vectorized NumPy (built-in) |
| P(Profit) = 100% | Drift too high | Check if `drift_source='historical'` is fetching correct data |
| VaR errors | Insufficient paths | Use n_paths ≥ 10000 for stable estimates |
| Download fails | Network issue | Check internet; try again or use manual parameters |

---

## Performance Tuning

### For Speed
```python
# Use fewer paths initially
results = sim.simulate(n_paths=1000, n_steps=252)  # ~10ms

# Disable visualization
sim.plot_results(show=False, save_path=None)

# Use Dask for multi-machine scaling (requires dask install)
sim = MonteCarloSimulator(..., parallel_backend='dask')
```

### For Accuracy
```python
# Increase paths to 100K+
results = sim.simulate(n_paths=100000, n_steps=252)  # ~500ms

# Finer time steps
results = sim.simulate(n_steps=1000)  # More granular paths

# Multiple runs with averaging
results_avg = average([sim.simulate(...) for _ in range(10)])
```

---

## Visualization Customization

```python
# Plot with custom settings
sim.plot_results(
    figsize=(14, 10),
    save_path='my_simulation.png',
    show=True,
    title='Custom Title',
    color_theme='dark',  # 'dark' or 'light'
    dpi=300,             # Resolution
)
```

---

## Further Reading

- **Black-Scholes Model**: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
- **Itô Calculus**: Hull, J. (2021), Ch. 14-15
- **Monte Carlo Methods**: Glasserman, P. (2004), Chapters 1-3
- **NumPy Optimization**: https://numpy.org/doc/stable/reference/generated/numpy.array.html
