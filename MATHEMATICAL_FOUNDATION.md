# Mathematical Foundation & Theory

## Table of Contents

1. [Geometric Brownian Motion](#geometric-brownian-motion)
2. [Itô Lemma & The σ²/2 Correction](#itô-lemma--the-σ22-correction)
3. [Discretization Schemes](#discretization-schemes)
4. [Statistical Properties](#statistical-properties)
5. [Convergence Analysis](#convergence-analysis)
6. [Distribution Theory](#distribution-theory)

---

## Geometric Brownian Motion

### Definition

The **Geometric Brownian Motion (GBM)** is the stochastic process defined by:

```
dS(t) = μS(t) dt + σS(t) dW(t)
```

Where:
- **S(t)** = Asset price at time t
- **μ** = Drift coefficient (expected instantaneous return)
- **σ** = Diffusion coefficient (volatility)
- **dW(t)** = Increment of standard Brownian motion
- **dt** = Infinitesimal time increment

### Properties

| Property | Value | Interpretation |
|----------|-------|-----------------|
| **E[S(t)]** | S₀ exp(μt) | Expected price grows exponentially |
| **Var[S(t)]** | S₀² exp(2μt)[exp(σ²t) - 1] | Variance increases with time & vol |
| **S(t) > 0** | Always | Prices never go negative (key feature) |
| **Continuous Paths** | Always | No sudden jumps (unrealistic for crises) |
| **Markovian** | Yes | Future depends only on current state |

### Why GBM?

✅ **Advantages**:
- Analytically tractable (closed-form solutions exist)
- Prevents negative prices
- Empirically reasonable for stocks (short-term)
- Foundation for Black-Scholes pricing

❌ **Disadvantages**:
- Misses jump events (market crashes)
- Underestimates tail risk (fat tails in reality)
- Assumes constant volatility (actually time-varying)
- Ignores transaction costs

---

## Itô Lemma & The σ²/2 Correction

### The Problem

If we naively apply standard calculus to solve dS = μS dt + σS dW, we get:

```
S(t) = S₀ exp(μt + σW(t))
```

But this **violates the rules of stochastic calculus!**

### Itô Lemma (Taylor Expansion in Stochastic Calculus)

For a function f(t, S) and SDE dS = a dt + b dW:

```
df = (∂f/∂t + a·∂f/∂S + (1/2)b²·∂²f/∂S²) dt + b·∂f/∂S dW
```

**Key difference from regular calculus**: The `(1/2)b²·∂²f/∂S²` term!

This arises because (dW)² = dt (non-zero), not (dW)² = 0.

### Application to GBM

Apply Itô Lemma to **f(S) = ln(S)**:

```
∂f/∂S = 1/S
∂²f/∂S² = -1/S²
∂f/∂t = 0
```

Substituting into Itô formula:

```
d(ln S) = [μ - σ²/2] dt + σ dW(t)
```

### The Closed-Form Solution

Integrating from t=0 to t=T:

```
ln(S(T)/S₀) = (μ - σ²/2)T + σ·W(T)
                           ↑
                    Itô correction!
```

Therefore:

```
S(T) = S₀ · exp[(μ - σ²/2)T + σ·√T·Z]
```

Where Z ~ N(0,1).

### Why σ²/2 Matters: A Numerical Example

**Setup**: S₀ = $100, μ = 10%, σ = 20%, T = 1 year

**Without Itô correction** (WRONG):
```
E[S(T)] = S₀ · exp(μT + σ√T·E[Z])
        = 100 · exp(0.10·1 + 0.20·1·0)
        = 100 · exp(0.10)
        = $110.52
```

**With Itô correction** (CORRECT):
```
E[S(T)] = S₀ · exp[(μ - σ²/2)T]
        = 100 · exp[(0.10 - 0.04/2)·1]
        = 100 · exp(0.08)
        = $108.33
```

**Difference**: $2.19 (1.96% error)

For a $1M portfolio: **$19,600 in valuation error!**

### Multi-Period Implications

The correction compounds. For 5-year options:

```
With correction:    S(5) = S₀ · exp[(0.10 - 0.02)·5 + ...] = $159.38
Without:           S(5) = S₀ · exp[0.10·5 + ...]          = $164.87
Cumulative error:  3.5%
```

---

## Discretization Schemes

### Euler-Maruyama (Exact for GBM)

For the SDE dS = μS dt + σS dW, the discrete approximation is:

```
S[n+1] = S[n] + μS[n]·Δt + σS[n]·√Δt·Z[n]
```

**Convergence**: O(Δt) weak error, O(√Δt) strong error

**Rearranged form** (what we implement):
```
S[n+1] = S[n] · exp[(μ - σ²/2)·Δt + σ·√Δt·Z[n]]
```

This is **exact** for GBM (not an approximation!).

### Milstein Scheme (Higher Order)

For non-GBM SDEs:

```
dS = a(S,t) dt + b(S,t) dW

S[n+1] = S[n] + a·Δt + b·√Δt·Z + (1/2)·b·∂b/∂S·(Z² - 1)·Δt
```

**Convergence**: O(Δt) strong error (vs O(√Δt) for Euler)

For GBM specifically, Milstein = Euler (no improvement).

### Why Euler is Sufficient

For GBM, the exact solution exists, and our log-space discretization is **exact** (not approximate):

```python
# Our implementation: EXACT solution
S[t+1] = S[t] * exp((mu - sigma**2/2)*dt + sigma*sqrt(dt)*Z[t])

# Error: O(Δt³), negligible for Δt = 1/252
```

---

## Statistical Properties

### Distribution of S(T)

**Theorem**: If S(0) = S₀ and dS = μS dt + σS dW, then:

```
S(T) ~ LogNormal(ln(S₀) + (μ - σ²/2)T, σ²T)
```

### Mean of Log-Normal Distribution

```
E[S(T)] = S₀ · exp(μT)
```

**Not** S₀ · exp[(μ - σ²/2)T]!

The σ²/2 appears in the **log**, not the mean.

### Median vs Mean

```
Median[S(T)] = S₀ · exp[(μ - σ²/2)T]    ← Median (smaller)
E[S(T)] = S₀ · exp(μT)                  ← Mean (larger)

Ratio = E/Median = exp(σ²T/2)
```

**Example**: σ = 20%, T = 1:
```
Ratio = exp(0.04) = 1.0408 (4.08% difference)
```

### Why Median < Mean

Log-normal distributions are **right-skewed**:
- A few very large values pull the mean upward
- Median better represents typical outcome
- Standard deviation scales with mean (proportional risk)

### Variance of S(T)

```
Var[S(T)] = S₀²·exp(2μT)·(exp(σ²T) - 1)
```

Increases both with:
1. **Time** (√T scaling for diffusion)
2. **Volatility** (exponential in σ²)

---

## Convergence Analysis

### Strong Convergence (Path-wise Error)

For Monte Carlo methods:

```
E[|S(T)_exact - S(T)_simulated|] = O(Δt^p)
```

Where p depends on the scheme:
- **Euler**: p = 1/2
- **Milstein**: p = 1
- **GBM exact formula**: p → ∞

### Weak Convergence (Expectation Error)

```
|E[S(T)_exact] - E[S(T)_simulated]| = O(Δt^q)
```

Where q depends on the scheme and test function.

### Statistical Convergence (n_paths)

For estimating E[f(S(T))]:

```
E[error] = O(1/√n_paths)
```

**Practical interpretation**:
- 100 paths: ~10% error
- 1,000 paths: ~3% error
- 10,000 paths: ~1% error
- 100,000 paths: ~0.3% error

This is the **law of large numbers**: 10× more paths → 3.16× better accuracy.

---

## Distribution Theory

### Parametric Family: Log-Normal

**Definition**: Y ~ LogNormal(μ, σ) if ln(Y) ~ Normal(μ, σ)

**PDF**:
```
f(x; μ, σ) = 1/(x·σ·√(2π)) · exp(-(ln(x) - μ)²/(2σ²))
```

**Moments**:
```
E[X] = exp(μ + σ²/2)
Var[X] = (exp(σ²) - 1)·exp(2μ + σ²)
Median = exp(μ)
Mode = exp(μ - σ²)
```

**Shape**: Right-skewed for all σ > 0

```
Skewness = (exp(σ²) + 2)·√(exp(σ²) - 1)
Kurtosis = exp(4σ²) + 2exp(3σ²) + 3exp(2σ²) - 6
```

### Tail Behavior

**Left tail** (downside risk):
```
P(X < x) ≈ Φ((ln(x) - μ)/(σ)) as x → 0
```

**Right tail** (upside):
```
P(X > x) ≈ Φ(-(ln(x) - μ)/(σ)) as x → ∞
```

Both tails decay as Gaussian (lighter than power-law).

### Fitting Log-Normal to Data

**Method 1: MLE (Maximum Likelihood)**

Given samples x₁, x₂, ..., xₙ:

```
μ̂ = (1/n)·Σ ln(xᵢ)
σ̂² = (1/n)·Σ (ln(xᵢ) - μ̂)²
```

**Method 2: Method of Moments**

Given E[X] and Var[X]:

```
σ̂² = ln(1 + Var[X]/E[X]²)
μ̂ = ln(E[X]) - σ̂²/2
```

---

## Jump-Diffusion Extensions (Merton Model)

For more realistic tail behavior, add Poisson jumps:

```
dS = μS dt + σS dW + (J - 1)S dN(t)
```

Where:
- **N(t)** = Poisson process (rare events)
- **J** = Jump size (multiplicative)
- **λ** = Jump intensity (avg jumps/year)

**Effect**: Creates discontinuities, fatter tails

**Implementation roadmap**: Can be added via parallel Poisson process

---

## Risk Metrics

### Value at Risk (VaR)

**Definition**: The loss amount at confidence level α:

```
VaR_α = -inf{x : P(Returns ≤ x) ≥ α}
```

For GBM:

```
VaR_α = S₀[exp(percentile_α(returns)) - 1]
```

**Example**: VaR(95%) = -15% means "95% confident won't lose more than 15%"

### Conditional VaR (Expected Shortfall)

**Definition**: Average loss conditioned on exceeding VaR:

```
CVaR_α = E[Returns | Returns ≤ VaR_α]
```

Always ≥ VaR (more conservative).

### Probability of Profit

```
P(Profit) = P(S(T) > S₀)
          = P(returns > 0)
          = 1 - Φ((μ - σ²/2)·T / (σ·√T))
```

**Simplifies to**:
```
P(Profit) = Φ(√T·(μ/σ - σ/2))
          = Φ(Sharpe·√T - σ/2·√T)
```

---

## Practical Validation

### Hypothesis Testing

**Test 1: Drift Recovery**

Claim: If we simulate with μ=0.10, can we recover it?

```
H₀: E[returns] = 0.10
H₁: E[returns] ≠ 0.10

Test statistic: t = (r̄ - 0.10) / (s/√n)
              ~ t(n-1) under H₀
```

With n=10K paths, we can detect drifts of ±1%.

**Test 2: Volatility Recovery**

```
H₀: σ = 0.20
H₁: σ ≠ 0.20

Test statistic: χ² = (n-1)·s²/σ₀²
              ~ χ²(n-1) under H₀
```

**Test 3: Log-Normal Fit**

Use Anderson-Darling test on ln(prices):

```
A² = -n - (1/n)·Σ(2i-1)·[ln(F(x_i)) + ln(1-F(x_n+1-i))]
```

Under H₀ (true log-normal): A² ~ AD distribution

---

## References

### Textbooks
1. **Hull, J.** (2021). Options, Futures, and Other Derivatives (11th ed). Pearson.
   - Chapters 14-15: Wiener Processes & Itô Lemma
2. **Glasserman, P.** (2004). Monte Carlo Methods in Financial Engineering. Springer.
   - Chapter 3: Brownian Motion & SDEs
3. **Shreve, S.** (2004). Stochastic Calculus for Finance II: Continuous-Time Models. Springer.
   - Complete rigorous treatment

### Papers
1. **Black, F., Scholes, M.** (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*, 81(3), 637-654.
2. **Merton, R.** (1973). "Theory of rational option pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183.

### Online Resources
- **MIT OpenCourseWare**: 18.S096 Topics in Mathematics with Applications in Finance
- **QuantInsti**: Itô Calculus Explained (visual guide)
- **NumPy Docs**: Random number generation and vectorization

