# Audit: `ferrotorch-distributions` vs `torch.distributions`

## Distribution coverage

| Distribution | ferrotorch | torch | Notes |
|---|---|---|---|
| Bernoulli | ✅ | ✅ | |
| Beta | ✅ | ✅ | |
| Binomial | ❌ | ✅ | gap |
| Categorical | ✅ | ✅ | |
| Cauchy | ✅ | ✅ | |
| Chi2 | ❌ | ✅ | gap |
| ContinuousBernoulli | ❌ | ✅ | gap |
| Dirichlet | ✅ | ✅ | |
| Exponential | ✅ | ✅ | |
| FisherSnedecor | ❌ | ✅ | gap |
| Gamma | ✅ | ✅ | |
| GeneralizedPareto | ❌ | ✅ | gap |
| Geometric | ❌ | ✅ | gap |
| Gumbel | ✅ | ✅ | |
| HalfCauchy | ❌ | ✅ | gap |
| HalfNormal | ✅ | ✅ | |
| Independent | ✅ | ✅ | |
| InverseGamma | ❌ | ✅ | gap |
| Kumaraswamy | ✅ | ✅ | |
| Laplace | ✅ | ✅ | |
| LKJCholesky | ❌ | ✅ | gap |
| LogNormal | ✅ | ✅ | |
| LogisticNormal | ❌ | ✅ | gap |
| LowRankMultivariateNormal | ✅ | ✅ | |
| MixtureSameFamily | ✅ | ✅ | |
| Multinomial | ✅ | ✅ | |
| MultivariateNormal | ✅ | ✅ | |
| NegativeBinomial | ❌ | ✅ | gap |
| Normal | ✅ | ✅ | |
| OneHotCategorical | ✅ | ✅ | |
| Pareto | ✅ | ✅ | |
| Poisson | ✅ | ✅ | |
| RelaxedBernoulli | ✅ | ✅ | |
| RelaxedOneHotCategorical | ✅ | ✅ | |
| StudentT | ✅ | ✅ | |
| TransformedDistribution | ✅ (in `transforms`) | ✅ | |
| Uniform | ✅ | ✅ | |
| VonMises | ✅ | ✅ | |
| Weibull | ✅ | ✅ | |
| Wishart | ❌ | ✅ | gap |

**Coverage: 25 of 35 distributions (~71%).**

## Infrastructure

| Component | ferrotorch | torch |
|---|---|---|
| `Distribution` base trait/class | (implicit per-distribution) | `distribution.Distribution` |
| `ExponentialFamily` base | ❌ | ✅ |
| `constraints` | ✅ | ✅ |
| `constraint_registry` | unclear | ✅ |
| `transforms` (bijective + log-det-Jac) | ✅ | ✅ |
| `kl` (analytic KL divergences) | ✅ | ✅ |
| `utils` | unclear | ✅ |

## API surface (per-distribution methods)

ferrotorch (per crate docstring):
- `sample` (no gradient)
- `rsample` (reparameterized, gradient flows)
- `log_prob`
- `entropy`

torch additionally has:
- `cdf(value)` — cumulative distribution function
- `icdf(value)` — inverse CDF (quantile)
- `enumerate_support()` — for discrete distributions, all support values
- `mean`, `mode`, `variance`, `stddev` — properties
- `sample_n(n)` — convenience for `sample(sample_shape=(n,))`
- `expand(batch_shape)` — broadcast distribution to a new batch shape
- `arg_constraints` — dict of parameter constraints
- `support` — constraint on the support set

**Gap:** likely missing `cdf`, `icdf`, `enumerate_support`, `expand`,
`arg_constraints`, statistical-moment properties.

## Recommendations

1. **Add the missing 10 distributions**, prioritized:
   - `Binomial`, `NegativeBinomial`, `Geometric` — common discrete
   - `Chi2`, `InverseGamma`, `FisherSnedecor`, `HalfCauchy` — Bayesian /
     MCMC use cases
   - `LKJCholesky`, `Wishart`, `LogisticNormal` — covariance / Stiefel
     priors
   - `ContinuousBernoulli` — VAE relaxation
   - `GeneralizedPareto` — extreme value modeling
2. **Add `cdf` / `icdf`** to all distributions where analytic forms exist
   (Normal, Cauchy, Exponential, Laplace, Gumbel, Uniform, Weibull,
   Logistic, Beta, Gamma).
3. **Add `mean`/`mode`/`variance`/`stddev`** properties.
4. **Add `expand(batch_shape)`** for distribution broadcasting (used in
   batched Bayesian inference).
5. **Add `enumerate_support`** for discrete distributions
   (Bernoulli, Categorical, Binomial, Geometric, NegativeBinomial,
   Poisson, OneHotCategorical, Multinomial).
6. **Add `ExponentialFamily` base** so users can implement custom
   exponential-family distributions without re-deriving infrastructure.
7. **Add `constraint_registry`** for registering custom transforms
   between constraints (used in `TransformedDistribution`).

## Status

**Coverage 71% by distribution count, 100% of the most-used
distributions** (Normal, Beta, Gamma, Bernoulli, Categorical, Dirichlet,
MultivariateNormal, MixtureSameFamily, etc.). Gaps are tail distributions
and methods (`cdf`/`icdf`/`mean`/`mode`).

**Do not split.** Maps cleanly to `torch.distributions`.
