Itô's lemma is what separates finance maths from standard calculus — the −σ22-\frac{\sigma^2}{2}
−2σ2​ correction is real and matters

Vectorised simulation with numpy is 100× faster than a Python loop
The median path always ends below the mean path under GBM — because log-normal distributions are right-skewed (a few enormous outcomes pull the mean up)
With 1,000 simulations the probability estimates (e.g. P(profit)) converge to within ~2% of the true values; with 10,000 they're very stable
Real stocks have fat tails — the 5th and 95th percentile outcomes happen more often than GBM predicts


Tech stack

yfinance — historical price data
numpy — vectorised GBM simulation
pandas — data handling and parameter estimation
matplotlib — 4-panel dark chart
scipy.stats — KDE for smooth distribution overlays
pytest — 20+ unit + statistical property tests




Periodic Daily Return=ln( 
Previous Day’s Price
Day’s Price
​
 )
​







Customizable Distributions: Support for Normal, Log-Normal, Uniform, and Poisson distributions.

Parallel Processing: Leverages multi-core CPUs for high-speed iterations.

Visual Analytics: Generates probability density functions (PDF) and cumulative distribution functions (CDF).

Sensitivity Analysis: Identify which input variables have the greatest impact on your results.
