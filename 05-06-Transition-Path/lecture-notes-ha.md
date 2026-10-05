---
title: '\textbf{How to solve a HA model?}'
subtitle: 'Advanced Macroeconomics: Heterogeneous Agent Models'
author: 'Raphaël Huleux'
date: '2026'
documentclass: article
fontsize: 12pt
fontfamily: mathpazo
linestretch: 1.5
geometry: margin=1in
numbersections: true
secnumdepth: 2
header-includes:
  - \usepackage{amsmath,amssymb,amsfonts}
  - \setlength{\parskip}{1.5ex plus 0.3ex minus 0.3ex}
  - \setlength{\parindent}{0pt}
  - \usepackage{caption}
  - \captionsetup{labelfont=bf}
  - \usepackage{float}
  - \makeatletter\def\fps@figure{H}\makeatother
  - \usepackage{enumitem}
  - \setlist[itemize,1]{label=--}
  - \setlist[itemize,2]{label=--}
  - \usepackage{titling}
  - \setlength{\droptitle}{-1.5cm}
  - \setlength{\emergencystretch}{1.5em}
  - \usepackage{tcolorbox}
  - \definecolor{DarkRed}{rgb}{0.7,0,0}
---

These notes give an overview of the methods of lectures 02 to 06.
The details, and the code, are in the notebooks; the figures are made in `lecture-notes-figs.ipynb`.

# Consumption-Saving

## The buffer-stock model

A household faces uninsurable income risk and a borrowing constraint, and chooses how much to consume and save.
For now, the interest rate $r$ and the wage $w$ are given (partial equilibrium).
In recursive form,

$$
\begin{aligned}
v(z_{it},a_{it-1}) &= \max_{c_{it}} u(c_{it}) + \beta \underline{v}(z_{it},a_{it}) \\
&\text{s.t.} \\
a_{it} &= (1+r)a_{it-1} + w z_{it} - c_{it} \geq 0 \\
\log z_{it+1} &= \rho_z \log z_{it} + \psi_{it+1}, \quad \mathbb{E}[z_{it}]=1
\end{aligned}
$$

where $\underline{v}(z_{it},a_{it}) = \mathbb{E}_t[v(z_{it+1},a_{it})]$ is the beginning-of-period value function.
The states are productivity $z_{it}$ and assets $a_{it-1}$, and the solution is a pair of policy functions $a^{\ast}(z_{it},a_{it-1})$ and $c^{\ast}(z_{it},a_{it-1})$.

The Euler equation holds with equality only when the constraint does not bind:

$$
u'(c_{it}) \geq \beta(1+r)\,\mathbb{E}_t\left[u'(c_{it+1})\right], \quad \text{with equality if } a_{it}>0
$$

- With prudence ($u'''>0$), income risk makes households save for precautionary reasons.
- The consumption function is concave in wealth, the MPC is high close to the constraint and falls with wealth, and households accumulate a buffer stock of savings, even with $\beta(1+r)<1$.
- As a consequence, who holds what matters: the aggregate MPC depends on the distribution of households.

Solving the household block for given $(r,w)$ takes two steps:

1. A **backward** step for the policy functions (EGM)
2. A **forward** step for the distribution (histogram method)

## Solving with EGM: the backward step

VFI puts an optimizer at every grid point, which is slow and inaccurate.
The endogenous grid method (EGM, Carroll 2006) iterates on the Euler equation instead.
The trick is to put the grid on the **choice** $a_t$ rather than on the state $a_{t-1}$, and ask: *which $a_{t-1}$ would make it optimal to choose this $a_t$?*

Start from a guess of the derivative of the beginning-of-period value function, $\underline{v}_a(z_t,a_t)$. Then

1. Invert the FOC: $c(z_t,a_t) = \left(\beta\,\underline{v}_a(z_t,a_t)\right)^{-1/\sigma}$
2. Endogenous grid from the budget constraint: $a_{t-1}(z_t,a_t) = \frac{a_t + c(z_t,a_t) - w z_t}{1+r}$
3. Interpolate to get $a^{\ast}(z_t,a_{t-1})$ back on the fixed grid, and set $a^{\ast}=0$ where it violates the borrowing constraint
4. Consumption from the budget constraint: $c^{\ast} = (1+r)a_{t-1} + w z_t - a^{\ast}$
5. Envelope condition and expectation: $\underline{v}_a = \Pi_z\left[(1+r)(c^{\ast})^{-\sigma}\right]$

One pass through steps 1-5 is **one backward step**: it maps $\underline{v}_{a,t+1}$ into $\underline{v}_{a,t}$ and the policies of period $t$.
For the infinite-horizon problem we repeat it until $\underline{v}_a$ stops changing.
No optimizer, one interpolation per $z$: much faster than VFI, and more accurate since we interpolate the policy function, which is close to linear.

This step is `solve_hh_backwards` in `GEModelTools`.

## Simulating with the histogram method: the forward step

Aggregates depend on how many households sit in each state.
Monte Carlo simulation works but is slow and noisy, and the noise puts a floor on how precisely we can clear markets.

The histogram method (Young, 2010) tracks **probability mass** instead of households.
The distribution is an array on the grids $\mathcal{G}_z\times\mathcal{G}_a$ that sums to one.
One period forward has two steps:

$$
\underline{\boldsymbol{D}}_t \overset{\Pi_z'}{\longrightarrow} \boldsymbol{D}_t \overset{\Lambda_t'}{\longrightarrow} \underline{\boldsymbol{D}}_{t+1}
$$

1. **Productivity draw:** $\boldsymbol{D}_t = \Pi_z'\underline{\boldsymbol{D}}_t$, the distribution over $(z_t,a_{t-1})$
2. **Savings lottery:** the mass at a grid point saves $a^{\ast}$, which falls between two grid points $a^i\leq a^{\ast}\leq a^{i+1}$. Send a share $\omega = \frac{a^{i+1}-a^{\ast}}{a^{i+1}-a^{i}}$ to $a^i$ and $1-\omega$ to $a^{i+1}$. The lottery has mean exactly $a^{\ast}$, so the grid does not distort aggregate savings.

Figure \ref{fig:histogram} follows the mass of one cell through both steps, on a toy grid.

![One period forward with the histogram method, on a toy grid with 3 productivity states and 7 asset points.](../02-03-Consumption-Saving/figs/sim_young_method.pdf){#fig:histogram width=100%}

Both maps are linear and deterministic: simulation is matrix algebra, no random draws.
For the stationary distribution we iterate until $\underline{\boldsymbol{D}}$ stops changing.
Aggregates are sums over the grid:

$$
A^{hh} = \sum \boldsymbol{D}\,a^{\ast}, \qquad C^{hh} = \sum \boldsymbol{D}\,c^{\ast}
$$

Two checks to always run: the distribution sums to one, and there is no mass at the top of the asset grid.

This step is `simulate_hh_forwards` in `GEModelTools`.

::: callout
- The buffer-stock model: income risk and a borrowing constraint give precautionary saving, a concave consumption function, and high MPCs at low wealth
- The solution is a pair of policy functions $a^{\ast}(z,a_{-1})$ and $c^{\ast}(z,a_{-1})$
- EGM (backward step): grid on the choice $a_t$, invert the Euler equation, endogenous grid from the budget constraint, interpolate back, impose the constraint
- Histogram method (forward step): the distribution is mass on the grid, moved by the productivity transition $\Pi_z'$ and the savings lottery $\Lambda'$
- Aggregates are sums of policies over the distribution, $A^{hh}=\sum\boldsymbol{D}\,a^{\ast}$
:::

# Stationary Equilibrium

## Prices are endogenous

In macro, $r$ and $w$ are not given: they are set by market clearing.
Households save more when $r$ is high, but their savings are the capital of the economy, and more capital lowers $r$.
To find the prices, we need the rest of the model.

## The HANC model

We complete the buffer-stock model into the Heterogeneous Agent Neo-Classical (HANC) model, also known as the Aiyagari model (Figure \ref{fig:blocks}):

1. **Firms:** rent capital from the mutual fund and hire labor from the households, produce with given technology, and sell output
2. **Zero-profit mutual fund:** owns the capital and rents it to firms, takes deposits and pays a return to households
3. **Households:** the buffer-stock model above, with labor supply $\ell_{it}=z_{it}$
4. **Markets:** perfect competition in the labor, goods and capital markets

![The HANC model as a set of blocks.](figs/notes/blocks.pdf){#fig:blocks width=75%}

Firms produce $Y_t=\Gamma_t K_{t-1}^{\alpha}L_t^{1-\alpha}$, where capital chosen in $t-1$ is used in $t$.
Profit maximization gives the prices:

$$
r^K_t = \alpha\Gamma_t\left(\frac{K_{t-1}}{L_t}\right)^{\alpha-1}, \qquad w_t = (1-\alpha)\Gamma_t\left(\frac{K_{t-1}}{L_t}\right)^{\alpha}
$$

The mutual fund has $A_t=K_t$ and pays $r_t=r^K_t-\delta$.
Markets clear:

$$
K_t=A^{hh}_t, \qquad L_t=\int z_{it}\,d\boldsymbol{D}_t=1, \qquad Y_t=C^{hh}_t+I_t
$$

By Walras' law, if the capital and labor markets clear, the goods market clears too: a free check on the solution.

## Definition

**Stationary equilibrium.** Given $\Gamma_{ss}$, quantities $K_{ss}$ and $L_{ss}$, prices $r_{ss}$ and $w_{ss}$, the distribution $\boldsymbol{D}_{ss}$ over $(z_{it},a_{it-1})$ and the policy functions $a^{\ast}_{ss}$ and $c^{\ast}_{ss}$ are such that

1. households maximize expected utility (policy functions),
2. firms maximize profits (prices),
3. $\boldsymbol{D}_{ss}$ is the invariant distribution implied by the household problem,
4. the mutual fund balance sheet is satisfied,
5. the capital, labor and goods markets clear.

Individual households keep moving inside the distribution, but the distribution itself, and hence all aggregates, are constant (law of large numbers).

## Solving it: a root-finding problem

With $L=1$, the firm conditions give $r$ and $w$ as functions of $K$:

$$
r(K) = \alpha\Gamma K^{\alpha-1}-\delta, \qquad w(K) = (1-\alpha)\Gamma K^{\alpha}
$$

The household block, solved backward and forward, gives aggregate savings $A^{hh}_{ss}(r,w)$.
The stationary equilibrium is then a one-dimensional **root-finding problem**:

$$
f(K) = A^{hh}_{ss}\big(r(K),w(K)\big) - K = 0
$$

1. Guess $K$
2. Compute $r(K)$ and $w(K)$ from the firm conditions
3. Solve the household problem backward (EGM) until convergence
4. Simulate the distribution forward (histogram) until convergence
5. Return $f(K) = A^{hh}_{ss}-K$, and update $K$ with a root-finder

Each evaluation of $f$ is a full backward and forward solve, so we want a root-finder that needs few evaluations.
There are other ways to set it up: guess $r$ instead of $K$, or fix $r$ and $K$ at their targets and find the $\beta$ that clears the market (calibration).

Note the difference with the representative agent: there, the Euler equation pins down $r_{ss}=1/\beta-1$ directly, and the asset supply curve is flat at that level. With heterogeneous agents no such equation exists, and precautionary saving pushes $r_{ss}$ below $1/\beta-1$.
In Figure \ref{fig:market-clearing}, households hold $K_{ss}=4$ at $r_{ss}=5\%$, where the representative agent would need $1/\beta-1=5.3\%$.

![Stationary equilibrium in HANC: household asset supply $A^{hh}_{ss}(r)$ against firm capital demand $K(r)$.](figs/notes/market_clearing.pdf){#fig:market-clearing width=65%}

## Root-finding algorithms: bisection and Newton

We look for $K^{\ast}$ such that $f(K^{\ast})=0$, where $f$ is continuous but has no closed form and is costly to evaluate.
Two classic methods trade off safety and speed.

**Bisection.** Start from a bracket $[K_{lo},K_{hi}]$ where $f(K_{lo})$ and $f(K_{hi})$ have opposite signs, so that a root lies in between.

1. Evaluate $f$ at the midpoint $K_{mid}=(K_{lo}+K_{hi})/2$
2. If $f(K_{mid})$ has the same sign as $f(K_{lo})$, set $K_{lo}=K_{mid}$, otherwise set $K_{hi}=K_{mid}$
3. Stop when $K_{hi}-K_{lo}$ is below a tolerance $\epsilon$

Each iteration halves the bracket, so we need about $\log_2\left((K_{hi}-K_{lo})/\epsilon\right)$ evaluations, e.g. 30 for a bracket of width 1 and $\epsilon=10^{-9}$.
Bisection always converges, and never leaves the bracket.
But it is slow, since it only uses the sign of $f$ and not its value, and it only works in one dimension.

**Newton.** Take a first-order approximation of $f$ around the current guess $K^n$ and set it to zero:

$$
f(K) \approx f(K^n) + f'(K^n)(K-K^n) = 0 \quad\Rightarrow\quad K^{n+1} = K^n - \frac{f(K^n)}{f'(K^n)}
$$

Close to the root, Newton converges quadratically: the number of correct digits roughly doubles with each iteration.
It has two drawbacks:

- It needs the derivative $f'$. In HANC there is no closed form, so we use a finite difference, $f'(K)\approx\left(f(K+h)-f(K)\right)/h$, which costs one more evaluation of $f$. The secant method saves this evaluation by using the slope between the last two guesses instead.
- It can fail far from the root. A step can overshoot to a $K$ so low that $r(K)\geq 1/\beta-1$, where household savings explode and $f$ is not even defined.

**In practice.** Brent's method (`scipy.optimize.brentq`) combines the two: it keeps a bracket like bisection, but takes faster interpolation steps whenever they stay inside the bracket.
It is as safe as bisection and almost as fast as Newton, and it is what we use for the steady state.
For the transition path the problem has $T$ unknowns and bisection no longer applies, but Newton does: the derivative $f'$ becomes a Jacobian matrix (see [Newton's method and the sequence-space Jacobian](#newtons-method-and-the-sequence-space-jacobian)).

::: callout
- In general equilibrium, prices are endogenous and set by market clearing
- HANC = buffer-stock households + firms + mutual fund + markets
- The definition of the stationary equilibrium
- Solving it is a one-dimensional root-finding problem: guess $K$, get $(r,w)$, solve backward, simulate forward, return $A^{hh}_{ss}-K$
- Bisection is safe but slow, Newton is fast but needs $f'$ and can diverge; `brentq` combines the two
- Precautionary saving pushes $r_{ss}$ below $1/\beta-1$
:::



# Transition Path

## Macro is about dynamics

The steady state tells us about the long run.
In macro we are interested in **dynamics**: how does the economy respond to a shock to TFP, to government spending, or to a change in policy?
A transition path is the path of the economy away from the steady state.

There are two types of transitions:

1. **Transitory:** the shock dies out, e.g. $\Gamma_t = \Gamma_{ss} + \sigma\rho^t$. The economy starts and ends in the **same** steady state. A transition can also start away from the steady state with no shock at all, e.g. after 5% of everyone's wealth is destroyed.
2. **Permanent:** the shock moves the economy to a **new** steady state, e.g. $\Gamma$ rises permanently by 5%. The economy starts in the old steady state and ends in the new one.

And two types of solutions:

1. **Nonlinear:** the exact path under perfect foresight.
2. **Linear:** a first-order approximation around the steady state. Much faster, precise for small shocks, but only for transitory shocks.

## The problem

Given initial conditions (the distribution $\underline{\boldsymbol{D}}_0$) and a path for the exogenous variables (TFP, parameters, policy), we look for the path of prices and endogenous variables such that, in every period, agents maximize and markets clear.

We make three assumptions:

1. **Perfect foresight:** at $t=0$, the shock is a surprise (an *MIT shock*). From then on, households know the whole future path of $\{r_t,w_t\}$.
2. **Truncation:** the economy is back at a steady state after $T$ periods, with $T$ large (e.g. $T=300$).
3. **Terminal condition:** we know the steady state we end in, including its $\underline{v}_a$.

**Transition path.** Given $\underline{\boldsymbol{D}}_0$ and a path $\{\Gamma_t\}$, quantities $\{K_t,L_t\}$, prices $\{r_t,w_t\}$, distributions $\{\boldsymbol{D}_t\}$ and policy functions $\{a^{\ast}_t,c^{\ast}_t\}$ are such that, in all periods,

1. households maximize expected utility, with perfect foresight over prices (policy functions),
2. firms maximize profits (prices),
3. $\boldsymbol{D}_t$ is implied by simulating the household problem forwards from $\underline{\boldsymbol{D}}_0$,
4. the mutual fund balance sheet is satisfied,
5. the capital, labor and goods markets clear.

## The equilibrium in sequence space

Write $\boldsymbol{X} = (X_0,X_1,\dots,X_{T-1})$ for the whole path of a variable.
In **sequence space**, we write whole paths of variables as functions of whole paths of other variables.

The household block is such a function.
Given a price path, we compute the aggregate response of households in two steps, as in the consumption-saving model, except that nothing is iterated to convergence and everything gets a $t$ subscript:

1. **Backward step:** start from the terminal condition $\underline{v}_{a,T} = \underline{v}_{a,ss}$ and apply one EGM step for each $t = T-1,\dots,0$, with the prices of period $t$. This gives time-varying policies $a_t^{\ast}$ and $c_t^{\ast}$.
2. **Forward step:** start from the initial condition $\underline{\boldsymbol{D}}_0$ and apply one histogram step for each $t = 0,\dots,T-1$, with the policies of period $t$. This gives $\boldsymbol{D}_t$.
3. **Aggregate:** $A^{hh}_t = \sum \boldsymbol{D}_t\,a^{\ast}_t$ for each $t$.

The result is the household impulse response, $\boldsymbol{A}^{hh}(\boldsymbol{r},\boldsymbol{w})$.
Perfect foresight is baked into the backward step: the policy at $t$ depends on all future prices through $\underline{v}_{a,t+1}$.

The firm block gives $\boldsymbol{r}(\boldsymbol{K},\boldsymbol{\Gamma})$ and $\boldsymbol{w}(\boldsymbol{K},\boldsymbol{\Gamma})$.
The whole equilibrium then reduces to $T$ equations in $T$ unknowns, asset market clearing in every period:

$$
\boldsymbol{H}(\boldsymbol{K},\boldsymbol{\Gamma}) = \boldsymbol{A}^{hh}\big(\boldsymbol{r}(\boldsymbol{K},\boldsymbol{\Gamma}),\boldsymbol{w}(\boldsymbol{K},\boldsymbol{\Gamma})\big) - \boldsymbol{K} = \boldsymbol{0}
$$

In `GEModelTools` language: **shocks** $\boldsymbol{Z}$ (here $\boldsymbol{\Gamma}$), **unknowns** $\boldsymbol{U}$ (here $\boldsymbol{K}$), **targets** $\boldsymbol{H}(\boldsymbol{U},\boldsymbol{Z})=\boldsymbol{0}$.
The DAG (directed acyclic graph) describes how to go from shocks and unknowns to targets: $\boldsymbol{K}\rightarrow$ firms $\rightarrow(\boldsymbol{r},\boldsymbol{w})\rightarrow$ households $\rightarrow\boldsymbol{A}^{hh}\rightarrow$ market clearing.

Why sequence space?

- The state of a HA economy includes the whole distribution $\boldsymbol{D}_t$: with $N_z=7$ and $N_a=500$, that is 3,500 state variables.
- In sequence space we never need the distribution as a state: under perfect foresight, households only need to know the price path.
- The unknowns are a few paths of aggregates ($T$ per unknown), whatever the size of the grids.

## Newton's method and the sequence-space Jacobian

$\boldsymbol{H}=\boldsymbol{0}$ is a root-finding problem again, but now multivariate: $T$ unknowns.
We solve it with **Newton's method**:

$$
\boldsymbol{K}^{n+1} = \boldsymbol{K}^n - \boldsymbol{H}_{\boldsymbol{K}}^{-1}\boldsymbol{H}(\boldsymbol{K}^n,\boldsymbol{\Gamma})
$$

This requires the Jacobian $\boldsymbol{H}_{\boldsymbol{K}}$, a $T\times T$ matrix: the **sequence-space Jacobian**.
We could compute $\boldsymbol{H}_{\boldsymbol{K}}$ directly, but it is more convenient to apply the chain rule along the DAG:

$$
\boldsymbol{H}_{\boldsymbol{K}} = \mathcal{J}^{A^{hh},r}\mathcal{J}^{r,K} + \mathcal{J}^{A^{hh},w}\mathcal{J}^{w,K} - \boldsymbol{I}
$$

The firm Jacobians $\mathcal{J}^{r,K}$ and $\mathcal{J}^{w,K}$ are cheap: differentiate the firm FOCs, and note that $r_t$ and $w_t$ depend on $K_{t-1}$, so they sit on the subdiagonal.
The household Jacobians are the expensive part.
Column $s$ of $\mathcal{J}^{A^{hh},r}$ is the household impulse response to a small change in $r_s$ only, so the naive approach needs $T$ impulse responses (see [Fake news algorithm](#fake-news-algorithm) for a faster way).

The Jacobian is not only a computational tool.
Figure \ref{fig:jacobian} plots three columns of $\mathcal{J}^{A^{hh},r}$ in HANC.
At $t=s$ savings jump: most of it is mechanical, the higher return on existing wealth ($A_{ss}=4$).
Before $s$ households already save more, in anticipation of the higher return.
After $s$ the extra wealth is consumed only slowly.

![Columns of the household Jacobian $\mathcal{J}^{A^{hh},r}$ in HANC, $T=300$.](figs/notes/jacobian.pdf){#fig:jacobian width=65%}

## The algorithm

**Nonlinear.** We compute $\boldsymbol{H}_{\boldsymbol{K}}$ **once**, around the steady state, and keep it fixed (quasi-Newton), or update it with Broyden's rule.
Each iteration then costs one household impulse response.

```
compute H_K around the steady state
guess K
repeat:
    r, w = firm(K, Gamma)
    A_hh = household impulse response to (r, w)
    H = A_hh - K
    stop if max|H| < tol
    K = K - solve(H_K, H)
```

Only the initial and terminal conditions differ between the two types of transitions:

- **Transitory:** $\underline{\boldsymbol{D}}_0 = \underline{\boldsymbol{D}}_{ss}$ and $\underline{v}_{a,T} = \underline{v}_{a,ss}$, and we guess $\boldsymbol{K}=K_{ss}$.
- **Permanent:** first solve for the new steady state. Then $\underline{\boldsymbol{D}}_0 = \underline{\boldsymbol{D}}_{ss}^{old}$ and $K_{-1} = K_{ss}^{old}$, but $\underline{v}_{a,T} = \underline{v}_{a,ss}^{new}$, and we guess $\boldsymbol{K}$ around $K_{ss}^{new}$. $T$ must be long enough for the economy to reach the new steady state.

In HANC, a transitory TFP shock with $\sigma=0.02$ and $\rho=0.9$ is solved in 5 Newton iterations.
Capital rises slowly, peaks about 10 years after the shock at $2.2\%$ above steady state, and then returns to it.
A permanent 5% rise in $\Gamma$ raises $K_{ss}$ from $4.00$ to $4.30$ and leaves $r_{ss}$ at $5\%$ (Figure \ref{fig:transitions}).
The interest rate does not move for a reason specific to this model: with a zero borrowing limit and income proportional to $w$, household savings scale with $w$, so capital and wages rise in proportion at the old $r$.

**Linear.** Take a first-order approximation of $\boldsymbol{H}(\boldsymbol{U},\boldsymbol{Z})=\boldsymbol{0}$ around the steady state:

$$
\boldsymbol{H}_{\boldsymbol{U}}d\boldsymbol{U} + \boldsymbol{H}_{\boldsymbol{Z}}d\boldsymbol{Z} = \boldsymbol{0} \quad\Leftrightarrow\quad d\boldsymbol{U} = \underbrace{-\boldsymbol{H}_{\boldsymbol{U}}^{-1}\boldsymbol{H}_{\boldsymbol{Z}}}_{\boldsymbol{G}}\,d\boldsymbol{Z}
$$

$\boldsymbol{H}_{\boldsymbol{Z}}$ is built with the chain rule, like $\boldsymbol{H}_{\boldsymbol{U}}$.
In HANC, $\boldsymbol{H}_{\boldsymbol{\Gamma}} = \mathcal{J}^{A^{hh},r}\mathcal{J}^{r,\Gamma} + \mathcal{J}^{A^{hh},w}\mathcal{J}^{w,\Gamma}$.
Once we have the Jacobians, the response to **any** shock path is one matrix product, $d\boldsymbol{K} = \boldsymbol{G}\,d\boldsymbol{\Gamma}$, and no household problem is solved.
For $\sigma=0.02$ (a 3% rise in TFP, since $\Gamma_{ss}=0.63$) the linear and nonlinear paths are indistinguishable; for $\sigma=0.2$ the linear path undershoots the peak of capital by about one percentage point.
A linear approximation around one steady state cannot move the economy to another, so permanent shocks require the nonlinear solution.

![Transitions in HANC. Left and middle: transitory TFP shocks with $\sigma=0.02$ and $\sigma=0.2$, nonlinear (Newton) and linear ($\boldsymbol{G}\,d\boldsymbol{\Gamma}$). Right: permanent 5% rise in TFP.](figs/notes/transitions.pdf){#fig:transitions width=100%}

Linear IRFs also let us **simulate** the economy.
Let $\text{IRF}^X_s$ be the effect on $X$ of a unit innovation $\epsilon$ to the shock, $s$ periods after it hit.
For an AR(1) TFP shock, it is the linear response to $d\Gamma_s=\rho^s$.
Three properties of the linear solution let us build any time series from this one IRF:

1. **Scaling:** an innovation of size $\epsilon_0$ moves $X$ by $\text{IRF}^X_s\,\epsilon_0$ after $s$ periods.
2. **Timing:** an innovation at date 1 has the same effect as one at date 0, one period later.
3. **Adding up:** the response to several innovations is the sum of the responses to each.

Hence

$$
dX_t = \sum_{s=0}^{T-1} \text{IRF}^X_s\,\epsilon_{t-s}
$$

where $s$ is the age of a shock: $\epsilon_{t-s}$ hit $s$ periods ago, and $\text{IRF}^X_s$ is how much a shock that old still moves $X$ today.
For example, with $\text{IRF}^X=(1,0.5,0.25,0,\dots)$ and innovations $\epsilon_0=1$ and $\epsilon_1=-2$, we get $dX_1 = 0.5\cdot 1 + 1\cdot(-2) = -1.5$.
To simulate, draw $\epsilon_t\sim\mathcal{N}(0,\sigma_\epsilon^2)$ and compute the sum, a convolution.
Moments follow without simulating: the innovations are independent, so $\text{var}(dX_t) = \sigma_\epsilon^2\sum_{s=0}^{T-1}(\text{IRF}^X_s)^2$.
The timing property holds because each innovation is a surprise when it hits, like an MIT shock (see [Aggregate risk and MIT shocks](#aggregate-risk-and-mit-shocks)).

In `GEModelTools`: `find_ss()`, `compute_jacs()`, then `find_transition_path()` (nonlinear), `find_IRFs()` (linear) and `simulate()`.

::: callout
- Transitory vs permanent transitions, and their initial and terminal conditions
- Nonlinear vs linear solutions
- Perfect foresight (MIT shocks) and truncation at $T$
- The equilibrium in sequence space: $\boldsymbol{H}(\boldsymbol{K},\boldsymbol{\Gamma})=\boldsymbol{A}^{hh}(\boldsymbol{r},\boldsymbol{w})-\boldsymbol{K}=\boldsymbol{0}$
- The household impulse response: backward from $\underline{v}_{a,T}$, forward from $\underline{\boldsymbol{D}}_0$
- Newton's method needs the sequence-space Jacobian, built with the chain rule along the DAG
- Linear solution: $d\boldsymbol{U}=\boldsymbol{G}\,d\boldsymbol{Z}$, and simulation as a sum of past shocks
:::

# Additional topics

## Aggregate risk and MIT shocks

MIT shocks assume that households never expect aggregate shocks.
That is hard to reconcile with business cycles, where shocks happen all the time.

But to **first order** it does not matter: the first-order solution to the perfect foresight transition (where households do not expect aggregate risk) is the same as the first-order solution of the model with aggregate risk.

To see this, take an Euler equation with a random aggregate variable $x_t = \rho x_{t-1} + \epsilon_t$, and linearize it around the deterministic steady state.
Only $d\mathbb{E}_t[x_{t+1}]$ shows up, and after an innovation $\epsilon_0$ it equals $\rho^{t+1}\epsilon_0$, the perfect-foresight path.
Certainty equivalence holds: the IRF to an MIT shock **is** the IRF of the model with aggregate risk, linearized in aggregate variables (Boppart, Krusell and Mitman, 2018).
This is why linear sequence-space IRFs can be simulated as if the model had aggregate risk.

At **second order**, a variance term $\sigma^2_{x,t}$ enters the Euler equation.
It is zero under perfect foresight but not with aggregate risk, so the two models differ.
What we miss with first-order methods: precautionary saving against aggregate risk, risk premia, and state dependence (the response to a shock does not depend on where the economy is, and scales linearly with the shock size).

## Fake news algorithm

The bottleneck is the household Jacobian, e.g. $\mathcal{J}^{A^{hh},w}$.
The naive approach shocks $w_s$ by a small $h$ for each $s\in\{0,\dots,T-1\}$ and computes a full impulse response each time.
Each column costs $T$ backward and $T$ forward steps: $T^2$ of each, for every household input.

The key insight of Auclert et al. (2021): **policy functions only depend on the distance to the shock**.
Let $\boldsymbol{y}_t^s$ be the policy at $t$ after a shock at $s$. Then

$$
\boldsymbol{y}_t^s = \begin{cases} \boldsymbol{y}_{T-1-s+t}^{T-1} & t\leq s \\ \boldsymbol{y}_{ss} & t>s \end{cases}
$$

After the shock has passed, households are back to their steady-state policies (their future is the steady state again).
Before it, only $s-t$ matters.

So we need **one** backward pass, for a shock at $s=T-1$, and we build the policies of every column by shifting it.
In the version coded in the notebook, we still do a forward simulation for each column ($T^2$ forward steps).
The full algorithm also avoids these, with a recursion through the so-called fake news matrix (the effect of news at $t=0$ about a shock at $s$ that is retracted at $t=1$).
The cost of the whole Jacobian is then about that of a single impulse response.
`GEModelTools` does all of this under the hood in `compute_jacs()`: we only declare the inputs and outputs of the household block.

## Models with perceived aggregate risk

Sometimes first order is not enough: we care about risk premia, precautionary saving against recessions, or large shocks with state dependence.
Then households must **perceive** aggregate risk, and the problem must be written in state-space form.
The canonical example is Krusell and Smith (1998), HANC with stochastic TFP:

$$
v(\boldsymbol{D}_t,\Gamma_t,z_{it},a_{it-1}) = \max_{c_{it}} u(c_{it}) + \beta\mathbb{E}_t\left[v(\boldsymbol{D}_{t+1},\Gamma_{t+1},z_{it+1},a_{it})\right]
$$

To forecast prices, households need to forecast $K_t = \int a_{it-1}\,d\boldsymbol{D}_t$, so the whole distribution $\boldsymbol{D}_t$ is a state variable.
The state space explodes.
Solutions in the literature:

- **Approximate aggregation:** households forecast prices with a few moments of $\boldsymbol{D}_t$ (Krusell and Smith, 1998).
- **State-space perturbation:** linearize (or go to higher order) in the distribution, e.g. Bayer and Luetticke (2020), Ahn et al. (2018). Harder to implement, but an easier path to second order.
- **Deep learning:** approximate the value or policy function in the distribution with neural networks, e.g. Fernández-Villaverde et al. (2021), Maliar et al. (2021).

In this course we stay in sequence space, with perfect foresight and first-order approximations.

::: callout
- To first order, the IRF to an MIT shock is the IRF of the model with aggregate risk (certainty equivalence)
- Fake news algorithm: policies only depend on the distance to the shock, so one backward pass gives the whole Jacobian
- With perceived aggregate risk, the distribution becomes a state variable: Krusell-Smith, state-space perturbation, deep learning
:::
