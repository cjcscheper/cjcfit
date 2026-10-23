# Theory of the fitting, the noise model and the Monte Carlo studies

This note explains what `cjcfit` computes and why, for students who have
fitted a line before but want to know where the uncertainties come from,
when to trust them, and what a Monte Carlo study adds.

Background: J. R. Taylor, *An Introduction to Error Analysis*, 2nd ed.,
University Science Books (1997), ch. 8 (straight line); P. R. Bevington &
D. K. Robinson, *Data Reduction and Error Analysis for the Physical
Sciences*, 3rd ed., McGraw-Hill (2003), ch. 6–8; W. H. Press et al.,
*Numerical Recipes*, 3rd ed., Cambridge (2007), ch. 15 (least squares,
the covariance, Monte Carlo of synthetic data sets); D. S. Sivia, *Data
Analysis: A Bayesian Tutorial*, 2nd ed., Oxford (2006) (priors). The
optimiser is scipy's trust-region reflective method: M. A. Branch,
T. F. Coleman & Y. Li, SIAM J. Sci. Comput. 21, 1 (1999).

## 1. What a fit does

A model $`y(x; \mathbf p)`$ with parameters $`\mathbf p = (p_1 \dots p_P)`$
is compared with $`N`$ measured values $`d_i`$ at $`x_i`$. The fit looks for
the $`\mathbf p`$ that makes the model closest to the data, and then asks
how far $`\mathbf p`$ could move before the model stops being compatible
with them: that is the uncertainty.

In cjcfit the model is any Python function `model(x, p_1=..., p_2=...)`;
`x` is passed through untouched (an array, a tuple of axes, anything), and
the parameters are given by name, so the fit and the model can never
disagree about their order.

## 2. Least squares and chi-squared

If each $`d_i`$ scatters around the true model with a Gaussian of standard
deviation $`\sigma_i`$, independently, the probability of the measured data
for given parameters is

```math
P(\mathbf d \mid \mathbf p) \propto \exp\!\left(-\tfrac12 \chi^2(\mathbf p)\right),
\qquad
\chi^2(\mathbf p) = \sum_{i=1}^{N} \left(\frac{d_i - y(x_i; \mathbf p)}{\sigma_i}\right)^2 .
```

The most probable parameters (maximum likelihood) are the ones with the
smallest $`\chi^2`$: least squares. `fit_least_squares` hands the vector of
normalized residuals $`r_i = (d_i - y_i)/\sigma_i`$ to
`scipy.optimize.least_squares`, which minimises $`\sum r_i^2`$ while
respecting the bounds.

## 3. Why the residuals are divided by sigma, and what weights add

Dividing by $`\sigma_i`$ makes every residual dimensionless and gives a
precise point more say than a noisy one: a point with half the
$`\sigma`$ counts four times as much. Without this weighting, a few noisy
points (the edges of a spectrum, where the signal is small) pull the fit as
hard as the good ones.

`weights` $`w_i`$ multiply the residuals the optimiser minimises,
$`\sum (w_i r_i)^2`$. They serve two purposes:

- **Without sigma**, as *relative* inverse sigmas, $`\sigma_i \propto 1/w_i`$:
  you know which points are noisier (e.g. from the spectrum's amplitude) but
  not the absolute noise level. Only the ratios of the weights matter.
- **With sigma**, as an extra emphasis beyond the noise: lowering a region
  you trust less than its $`\sigma`$ says (a model known to be rough there),
  or $`w_i = 0`$ to leave a point out (an outlier, a band outside the usable
  range) without cutting the arrays.

A weight 0 also removes the point from $`\chi^2`$ and from the degrees of
freedom.

With the right $`\sigma_i`$ and the right model, each $`r_i`$ is a standard
normal number, so $`\chi^2`$ is about the number of degrees of freedom
$`\nu = N - P`$ (the $`P`$ fitted parameters absorb $`P`$ of them):

```math
\chi^2_\mathrm{red} = \frac{\chi^2}{\nu} \approx 1 \pm \sqrt{2/\nu}.
```

Well above 1: the model misses something, or the $`\sigma_i`$ are too small.
Well below 1: the $`\sigma_i`$ are too large. A plot of $`r_i`$ against
$`x_i`$ shows which: structure in the residuals points at the model, a
uniform excess at sigma.

## 4. Uncertainties from the Jacobian

Near the minimum $`\hat{\mathbf p}`$ the model is close to linear in the
parameters, $`y(\mathbf p) \approx y(\hat{\mathbf p}) + J (\mathbf p - \hat{\mathbf p})`$,
with the Jacobian of the normalized residuals
$`J_{ik} = \partial r_i / \partial p_k`$. Then $`\chi^2`$ is a quadratic
bowl, $`\chi^2(\mathbf p) \approx \chi^2_\mathrm{min} + (\mathbf p - \hat{\mathbf p})^T J^T J (\mathbf p - \hat{\mathbf p})`$,
and the parameters have the covariance

```math
C = (J^T J)^{-1}, \qquad \sigma_{p_k} = \sqrt{C_{kk}}, \qquad
\rho_{kl} = \frac{C_{kl}}{\sigma_{p_k}\sigma_{p_l}} .
```

For a straight line this is exactly the textbook result (Taylor, ch. 8):
with $`\Delta = N\sum x_i^2 - (\sum x_i)^2`$,
$`\sigma_\mathrm{slope} = \sigma\sqrt{N/\Delta}`$ and
$`\sigma_\mathrm{intercept} = \sigma\sqrt{\sum x_i^2/\Delta}`$
(`test_line_uncertainties_equal_the_textbook_formulas`). A correlation
$`\rho`$ near $`\pm 1`$ means two parameters can compensate each other: the
data determine a combination of them better than either alone.

**With and without sigma.** With `sigma` given, $`C`$ is used as it is: the
uncertainties follow from the $`\sigma_i`$ you trust, and
$`\chi^2_\mathrm{red}`$ tests them. Without `sigma`, every point gets
$`\sigma_i = 1`$ in the fit, and $`C`$ is multiplied by $`\chi^2/\nu`$: the
common $`\sigma`$ is estimated from the scatter of the residuals. That is
right when all points have the same (unknown) noise and the model fits;
it hides a wrong model by inflating the uncertainties. Weights alone
($`\sigma_i = c/w_i`$) are handled the same way, with $`c`$ from the scatter.

**Sigma and weights together.** The fit then minimises
$`\sum (w_i r_i)^2`$, which is not the maximum-likelihood fit for the given
$`\sigma_i`$, and $`(J_w^T J_w)^{-1}`$ of the weighted Jacobian $`J_w`$ is
not its covariance. Propagating the noise of the data (the weighted residuals
have variance $`w_i^2`$, a prior residual 1) through the linearised fit gives
the "sandwich"

```math
C = H^{-1}\, J_w^T \,\mathrm{diag}(w_i^2)\, J_w\, H^{-1}, \qquad H = J_w^T J_w ,
```

which reduces to $`H^{-1}`$ for $`w_i = 1`$. Weights that differ from the
noise's own $`1/\sigma_i`$ always give a somewhat *larger* uncertainty than
the unweighted $`\sigma`$-fit: that is the price of the emphasis
(`test_sigma_and_weights_give_the_sandwich_covariance` checks it against the
weighted straight line by hand). $`\chi^2`$ is then computed without the
weights, so $`\chi^2_\mathrm{red}`$ still tests the $`\sigma_i`$.

**When to trust it.** The linearisation holds when the model is close to
linear over $`\pm\sigma_p`$ around the optimum. It fails

- when the noise is large compared with the curvature of the model (a
  width that is barely resolved, a phase that wraps);
- at a bound: the true optimum may lie beyond it (`FitResult.at_bound`,
  `parameters_at_bound`);
- when the data do not determine a parameter at all: $`J`$ has a zero
  singular value and $`J^TJ`$ cannot be inverted. cjcfit then reports
  infinite uncertainties and warns, instead of a pseudo-inverse's misleading
  zero.

A Monte Carlo study (section 9) tests the linearisation directly.

The Jacobian is computed by finite differences. The default step (about
$`10^{-8}`$ relative) differentiates rounding errors when the model itself
has numerical noise (a PDE solver, an interpolated table); `diff_step` sets
an absolute step, larger than that noise and small compared with
$`\sigma_p`$ (`suggest_diff_step` gives 0.1 % of the value).

## 5. Parameter scaling

The optimiser takes steps in parameter space and stops when the steps or the
changes of $`\chi^2`$ fall below tolerances that assume parameters of order
one. A thickness of $`5\times10^{-4}`$ m or a carrier density of
$`10^{22}`$ m⁻³ violates that. `scale` changes the space the optimiser
works in, without changing anything the user sees:

- `"normalized"`: $`q = p / s`$ with `normalization_factor` $`s`$
  (a thickness in µm, $`s = 10^{-6}`$);
- `"log10"` (or `"ln"`): $`q = \log_{10} p`$. A step in $`q`$ is a *factor*
  in $`p`$. This is the right space for a positive quantity that may lie
  anywhere over several decades: going from $`10^3`$ to $`10^6`$ is a
  distance of 3, and the model often depends on $`\log p`$ more evenly than
  on $`p`$ (`test_log10_scale_reaches_a_parameter_decades_away`: a decay rate
  four decades off is found in log10).

Bounds, priors, steps and start spreads are converted to $`q`$ internally.
The covariance is converted back to physical units with
$`\partial p/\partial q`$ ($`= p \ln 10`$ for log10):
$`C_p = D\,C_q\,D`$, $`D = \mathrm{diag}(\partial p_k/\partial q_k)`$. To
first order the reported uncertainty does not depend on the scale
(`test_uncertainty_in_physical_units_does_not_depend_on_the_scale`).

## 6. Priors as extra residuals

Prior knowledge "$`p_k = \mu \pm s`$" (a thickness measured with a
micrometer, a literature value) multiplies the likelihood by a Gaussian,
$`\exp(-\tfrac12 ((p_k - \mu)/s)^2)`$. Maximising the product is the same
as least squares with one extra residual

```math
r_{N+1} = \frac{p_k - \mu}{s},
```

so the prior acts as one more "measurement" of $`p_k`$. For a parameter
that the data measure as $`\hat p \pm \sigma_d`$, the result is the
inverse-variance weighted mean of data and prior,
$`(\hat p/\sigma_d^2 + \mu/s^2)/(1/\sigma_d^2 + 1/s^2)`$, with uncertainty
$`(1/\sigma_d^2 + 1/s^2)^{-1/2}`$
(`test_prior_combines_with_the_data_as_inverse_variance_weights`). For a
parameter the data do not determine at all, the prior alone sets it.

The prior is in physical units, so the data residuals must be too: give
`sigma` when using priors. For a log scale the prior is Gaussian in
$`\log p`$ with the spread converted to first order,
$`s_q = s/(\mu \ln 10)`$ (a log-normal prior). `FitResult.cost` is
$`\chi^2`$ plus the prior terms; `chi_squared` and its degrees of freedom
count the data only.

## 7. Complex data

A complex measurement (a field ratio $`t(f)`$, an impedance) is two real
measurements per point. cjcfit compares Re and Im separately and stacks
them into one residual vector

```math
\mathbf r = \left[\frac{\mathrm{Re}(d_i - y_i)}{\sigma_{\mathrm{re},i}} \dots,\;
\frac{\mathrm{Im}(d_i - y_i)}{\sigma_{\mathrm{im},i}} \dots\right],
```

so $`\chi^2 = \sum |r|^2`$ over $`2N`$ real values and $`\nu = 2N - P`$.
Fitting Re and Im (rather than magnitude and phase) keeps the residuals
Gaussian when the noise is additive, and avoids phase unwrapping. A real
`sigma` applies to both parts; a complex `sigma` is
$`\sigma_\mathrm{re} + i\sigma_\mathrm{im}`$. Example: for
$`A e^{-i 2\pi f\tau}`$ with the same $`\sigma`$ on Re and Im,
$`\sigma_A = \sigma/\sqrt N`$ and
$`\sigma_\tau = \sigma/(2\pi A \sqrt{\sum f_i^2})`$, uncorrelated
(`test_complex_amplitude_and_delay`).

## 8. The noise model

Pseudo-measurements need noise that looks like a real instrument's. Two
mechanisms cover most cases, and `add_noise` combines them:

```math
y_\mathrm{noisy} = y\,(1 + a\,n_1)\,e^{i b n_2} \;+\; (s_\mathrm{re}\,n_3 + i\,s_\mathrm{im}\,n_4),
```

with independent standard normal numbers $`n_1 \dots n_4`$ per point.

- **Additive** ($`s_\mathrm{re} + i s_\mathrm{im}`$): noise that does not
  depend on the signal: detector and amplifier noise, a noise floor. A real
  value means the same $`s`$ on Re and Im.
- **Multiplicative** ($`a + i b`$): noise proportional to the signal. The real
  part $`a`$ is a relative amplitude noise (laser intensity noise), the
  imaginary part $`b`$ a phase noise in radians (timing jitter
  $`\sigma_\tau`$ gives $`b = 2\pi f \sigma_\tau`$). It is applied exactly,
  as a scaling and a rotation, not linearised, so it stays correct for large
  phase noise.

Each of $`a, b, s`$ may be a number, an array (one value per point, i.e.
dispersive noise) or a function of x such as a "bathtub" in frequency
(`examples/example_noise_models.py`, `example_lab_c_noise.py`). For real data
only $`a`$ and $`s_\mathrm{re}`$ exist; an imaginary part is an error.

`noise_sigma` gives the standard deviation of Re and Im to first order in
$`a, b`$: from $`\delta y = y(a n_1 + i b n_2)`$,

```math
\sigma_\mathrm{re}^2 = (a\,\mathrm{Re}\,y)^2 + (b\,\mathrm{Im}\,y)^2 + s_\mathrm{re}^2, \qquad
\sigma_\mathrm{im}^2 = (a\,\mathrm{Im}\,y)^2 + (b\,\mathrm{Re}\,y)^2 + s_\mathrm{im}^2 .
```

For $`a \ne b`$ the Re and Im noise of one point are correlated, which a
per-point sigma does not describe; as a weight it is still the right size.

**Phase noise biases the amplitude of a complex fit.** A random phase
rotation does not change $`|y|`$, but on average it pulls the complex value
towards the origin: $`\langle e^{i b n} \rangle = e^{-b^2/2}`$. A fit of Re
and Im finds that average, so the fitted amplitude comes out too small by
the factor $`e^{-b^2/2}`$: 0.5 % for $`b = 0.1`$ rad, 4.4 % for $`b = 0.3`$
rad. (`examples/example_complex_data.py` therefore uses additive noise, which
a Re/Im fit recovers without bias.) For the phase noise of a real
setup (a few hundredths of a radian) this bias is far below the
uncertainty; for larger $`b`$, check it with `monte_carlo_noise_fit`.

## 9. Monte Carlo: two different questions

**`monte_carlo_noise_fit`: how precise and how biased is the fit?**
Generate the model at known true values, add fresh noise, fit, repeat
(Press et al., section 15.6, "synthetic data sets"). The distribution of the
fitted values is the actual sampling distribution of the estimator for this
model, noise and start: its standard deviation is the real uncertainty, its
mean minus the truth the bias. Comparing the spread with the median of the
uncertainties the fits *report* tests section 4: they agree when the
linearisation holds (`examples/example_monte_carlo.py`, study 2); a spread much
larger than the reported uncertainty, a skewed histogram or many fits at a
bound show where it does not. This is the tool for questions like "can this
measurement distinguish the thickness from the carrier density at this
noise level?", before measuring. Every run starts at the parameters'
`initial` values, so the result describes the noise alone. With
`random_start=True` every run starts at its own `initial ± initial_std`
instead, as a real measurement is fitted from a start that is only roughly
right; runs that fall into another minimum then appear in the spread, the
bias and the failed fits. The noise of each run is the same in both cases
(the starts come from a separate set of seeds), so comparing the two shows
exactly what the start values add.

**`monte_carlo_convergence_fit`: does the fit find the same minimum?**
Fit the *same* data many times from start values drawn around `initial`
with spread `initial_std`. The data never change, so the spread of the end
points is not an uncertainty. It maps the cost landscape: one cluster of
end points with the same lowest cost means the fit is robust over that range
of starts; several clusters mean several minima, and the start values decide
which one is found (`examples/example_monte_carlo.py`, study 1). The run
with the lowest cost is the best fit.

Both run their fits in separate processes: a fit is pure Python and holds
the interpreter lock, so threads share one core. Each run gets its own seed,
spawned from the study's seed (`numpy.random.SeedSequence`), so the results
do not depend on the number of processes.

## 10. A grid as a starting-point search

Least squares only walks downhill. When $`\chi^2`$ has several valleys (a
frequency or a thickness that can trade whole periods of an oscillation, a
parameter over many decades), the start decides the valley.
`discrete_grid_fit` evaluates $`\chi^2`$ at every combination of given
values: no optimisation, just a map. Its lowest point is in (or next to) the
right valley and is a safe start for the least-squares fit
(`initial_values=grid.best_values`, `examples/example_grid_fit.py`). The
map also shows the shape of the minimum: long diagonal valleys are strongly
correlated parameters. `cjcfit.app.pick_initial_values` shows the same map
for two parameters and lets you click the start point. The cost grows as
the product of the grid sizes, so keep grids to two or three parameters and
use log-spaced values for parameters over decades.
