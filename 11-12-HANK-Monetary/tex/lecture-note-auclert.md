---
title: '\textbf{Auclert (2019) in the canonical HANK model}'
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

This note derives Auclert's (2019) decomposition of the impact response of consumption from the monetary IKC of part 1.
The intuition for each channel is in the slides.
Section 1 decomposes the Jacobian of a transfer, section 2 derives the monetary IKC, and section 3 derives Auclert's decomposition on impact.

# A Jacobian decomposition

Let $MPC_{i,t,s}=\partial c_{it}/\partial y_{is}$, and write $MPC_i\equiv MPC_{i,0,0}$.
Expectations and covariances are over households in the steady state.

A transfer $dx_s$ at date $s$ pays $\omega_{is}\,dx_s$ to household $i$, with $\mathbb{E}[\omega_{is}]=1$.
Since $\mathbb{E}[XY]=\mathbb{E}[X]\,\mathbb{E}[Y]+\text{cov}(X,Y)$,
$$\frac{\partial C_t}{\partial x_s}=\mathbb{E}\big[\omega_{is}\,MPC_{i,t,s}\big]=\underbrace{\mathbb{E}\big[MPC_{i,t,s}\big]}_{M^\chi_{t,s}}+\text{cov}\big(\omega_{is},MPC_{i,t,s}\big)$$
where $\boldsymbol{M}^\chi$ is the Jacobian of a lump-sum transfer.
Two transfers of the same aggregate size therefore differ only by a covariance.

# From the monetary IKC to Auclert's channels

## Model

This is part 1's full model with Auclert's fiscal rule.
Tax changes are lump-sum, and the government rebates its gains immediately.
The rate $r_t$ runs from $t$ to $t+1$, and $r^p_t$ is the realized return on wealth from $t-1$ to $t$.

$$
\begin{aligned}
\text{Households:}\quad & c_{it}+a_{it}=(1+r^p_t)\,a_{i,t-1}+Z_t\frac{z_{it}^{1+\zeta\log(Z_t/Z_{ss})}}{\mathbb{E}\big[z_{it}^{1+\zeta\log(Z_t/Z_{ss})}\big]}-T_t,\qquad a_{it}\ge0\\
\text{Firms:}\quad & Y_t=L_t,\qquad Z_t=\tfrac{1}{\mu}Y_t,\qquad d_t=\left(1-\tfrac{1}{\mu}\right)Y_t\\
\text{Stocks:}\quad & p_t=\frac{p_{t+1}+d_{t+1}}{1+r_t}\\
\text{Wealth:}\quad & A_t=p_t+B_t,\qquad (1+r^p_t)\,A_{t-1}=p_t+d_t+(1+r^b_t)\,B_{t-1}\\
\text{Nominal bonds:}\quad & 1+r^b_t=1+r_{t-1}\text{ for }t\ge1,\qquad dr^b_0\approx-(1+r_{ss})\,d\pi_0\\
\text{Government:}\quad & T_t+B_t=G_{ss}+(1+r^b_t)\,B_{t-1},\qquad B_t=\frac{(1+r_{ss})B_{ss}}{1+r_t}\\
\text{Central bank:}\quad & r_t=r_{ss}+\epsilon^r_t\\
\text{Market clearing:}\quad & Y_t=C_t+G_{ss},\qquad A_t=\textstyle\int a_{it}\,di
\end{aligned}
$$

*Unequal incidence.* Differentiating household $i$'s share of labor income at the steady state gives
$$dz_{it}=\zeta\,z_{it}\big(\log z_{it}-\mathbb{E}[z\log z]\big)\frac{dZ_t}{Z_{ss}},\qquad\mathbb{E}[dz_{it}]=0$$
so household $i$'s labor income moves by $z_{it}\,dZ_t+Z_{ss}\,dz_{it}$.

*Taxes.* After $t=0$, interest payments are constant, $(1+r^b_t)B_{t-1}=(1+r_{ss})B_{ss}$.
Linearizing the government budget with $dB_t=-\frac{B_{ss}}{1+r_{ss}}dr_t$ gives
$$dT_0=B_{ss}\,dr^b_0+\frac{B_{ss}}{1+r_{ss}}dr_0,\qquad dT_t=\frac{B_{ss}}{1+r_{ss}}dr_t\ \text{ for }t\ge1$$

*Return on wealth.* As in part 1, $dr^p_t=dr_{t-1}$ for $t\ge1$, and $A_{ss}\,dr^p_0=\text{cap}_0+B_{ss}\,dr^b_0$ with
$$\text{cap}_0=\left(1-\tfrac{1}{\mu}\right)\boldsymbol{q}'d\boldsymbol{Y}-\tfrac{p}{1+r_{ss}}\,\boldsymbol{q}'d\boldsymbol{r}$$

## Monetary IKC

Households take as given labor income $\boldsymbol{Z}$, taxes $\boldsymbol{T}$, the rate $\boldsymbol{r}$, and the surprise return $dr^p_0$.
By section 1, the Jacobian of each transfer is $\boldsymbol{M}^\chi$ plus a covariance, with weights:

- $\boldsymbol{M}^\chi$: lump-sum taxes, weights $1$. Its first column is $\boldsymbol{m}^\chi$.
- $\boldsymbol{M}$: labor income with $\zeta=0$, weights $z_{is}$.
- $\tilde{\boldsymbol{M}}$: labor income with unequal incidence, weights $z_{is}+Z_{ss}\,\partial z_{is}/\partial Z_s$.
- $\boldsymbol{m}$: a payment at $t=0$ in proportion to wealth, weights $a_{i,-1}/A_{ss}$.

$\boldsymbol{M}^r$ is part 1's Jacobian of the rate.
It is not a transfer.
A higher $r_s$ pays $a_{is}\,dr_s$ to household $i$ at $s+1$ (income effect), and enters the Euler equation at $s$ (substitution effect).

**Step 1: linearize.** With $dZ_t=\frac{1}{\mu}dY_t$,
$$d\boldsymbol{C}=\tfrac{1}{\mu}\tilde{\boldsymbol{M}}\,d\boldsymbol{Y}-\boldsymbol{M}^\chi d\boldsymbol{T}+\boldsymbol{m}\,\big(\text{cap}_0+B_{ss}dr^b_0\big)+\boldsymbol{M}^r d\boldsymbol{r}$$

**Step 2: substitute** $\text{cap}_0$ and $d\boldsymbol{T}$, and set $d\boldsymbol{Y}=d\boldsymbol{C}$:
$$
\begin{aligned}
d\boldsymbol{Y}=\;&\tfrac{1}{\mu}\tilde{\boldsymbol{M}}\,d\boldsymbol{Y}+\left(1-\tfrac{1}{\mu}\right)\boldsymbol{m}\boldsymbol{q}'d\boldsymbol{Y}+B_{ss}\boldsymbol{m}\,dr^b_0-B_{ss}\boldsymbol{m}^\chi dr^b_0\\
&+\boldsymbol{M}^r d\boldsymbol{r}-\tfrac{p}{1+r_{ss}}\boldsymbol{m}\boldsymbol{q}'d\boldsymbol{r}-\tfrac{B_{ss}}{1+r_{ss}}\boldsymbol{M}^\chi d\boldsymbol{r}
\end{aligned}
$$

**Step 3: group by shock.** Write $\tilde{\boldsymbol{M}}=\boldsymbol{M}+(\tilde{\boldsymbol{M}}-\boldsymbol{M})$ and $\bar{\boldsymbol{M}}=\frac{1}{\mu}\boldsymbol{M}+\left(1-\frac{1}{\mu}\right)\boldsymbol{m}\boldsymbol{q}'$:
$$
\begin{aligned}
d\boldsymbol{Y}=\;&\underbrace{\bar{\boldsymbol{M}}\,d\boldsymbol{Y}}_{\text{aggregate income}}+\underbrace{\tfrac{1}{\mu}(\tilde{\boldsymbol{M}}-\boldsymbol{M})\,d\boldsymbol{Y}}_{\text{earnings heterogeneity}}+\underbrace{B_{ss}\big(\boldsymbol{m}-\boldsymbol{m}^\chi\big)dr^b_0}_{\text{Fisher}}\\[1ex]
&+\underbrace{\Big(\boldsymbol{M}^r-\tfrac{1}{1+r_{ss}}\big(p\,\boldsymbol{m}\boldsymbol{q}'+B_{ss}\boldsymbol{M}^\chi\big)\Big)d\boldsymbol{r}}_{\text{interest rate exposure and substitution}}
\end{aligned}
$$

Earnings heterogeneity and Fisher compare two transfers of the same aggregate size, so each is a covariance.
In the last term, the income effect of $\boldsymbol{M}^r$, the capital gain and the tax cut sum to $\frac{A_{ss}-p-B_{ss}}{1+r_{ss}}dr_s=0$ in present value, so they are a redistribution, the interest rate exposure channel.
The remaining term is the substitution effect of $\boldsymbol{M}^r$.

*RANK.* With one household, $\boldsymbol{M}^\chi=\bar{\boldsymbol{M}}=\boldsymbol{m}\boldsymbol{q}'$, so earnings heterogeneity and Fisher vanish.
The income effect of $\boldsymbol{M}^r$ is $\frac{A_{ss}}{1+r_{ss}}\boldsymbol{m}\boldsymbol{q}'$, which the capital gain and the tax cut offset since $p+B_{ss}=A_{ss}$.
Only aggregate income and substitution remain.

# Auclert's decomposition of $dC_0$

## Result

For a one-period rate change $dr_0$, a one-period change in output $dY_0$ and a surprise $d\pi_0$, the first row of the IKC is
$$
\begin{aligned}
dC_0=\;&\underbrace{\mathbb{E}\Big[\tfrac{Y_i}{Y}MPC_i\Big]dY_0}_{\text{aggregate income}}+\underbrace{\text{cov}\big(MPC_i,\;Z_{ss}\,dz_{i0}\big)}_{\text{earnings heterogeneity}}-\underbrace{\text{cov}\big(MPC_i,\,NNP_i\big)\,d\pi_0}_{\text{Fisher}}\\[1ex]
&+\Big[\underbrace{\text{cov}\big(MPC_i,\,URE_i\big)}_{\text{interest rate exposure}}-\underbrace{\tfrac{1}{\sigma}\,\mathbb{E}\big[(1-MPC_i)\,c_{i0}\big]}_{\text{substitution}}\Big]\frac{dr_0}{1+r_{ss}}
\end{aligned}
$$
with the household exposures

- income share: $\frac{Y_i}{Y}=\frac{1}{\mu}z_{i0}+\left(1-\frac{1}{\mu}\right)\frac{a_{i,-1}}{A_{ss}}$,
- net nominal position: $NNP_i=(1+r_{ss})\frac{B_{ss}}{A_{ss}}a_{i,-1}$,
- unhedged interest rate exposure: $URE_i=a_{i0}-\frac{p}{A_{ss}}a_{i,-1}$.

In Auclert's notation, $\frac{dr_0}{1+r_{ss}}$ is $\frac{dR}{R}$, $d\pi_0$ is $\frac{dP}{P}$, and his EIS $\sigma$ is our $\frac{1}{\sigma}$.

## Two facts

The derivation uses two results, proved in the appendix.

*Fact 1.* An unconstrained household ($a_{i0}>0$) has $MPC_{i,0,1}=\frac{MPC_i}{1+r_{ss}}$.
A constrained household has $MPC_{i,0,1}=0$.

*Fact 2.* At unchanged income, a change in the rate in the Euler equation moves consumption by
$$dc_{i0}=-\frac{1}{\sigma}(1-MPC_i)\,c_{i0}\frac{dr_0}{1+r_{ss}}$$
for all households (Auclert's Theorem 2).

## Derivation

**Step 0.** With $d\boldsymbol{r}=(dr_0,0,\dots)'$ and $d\boldsymbol{Y}=(dY_0,0,\dots)'$, row 0 of the IKC uses only the first entry of each matrix, and $\boldsymbol{q}'d\boldsymbol{Y}=dY_0$, $\boldsymbol{q}'d\boldsymbol{r}=dr_0$.

**Step 1: aggregate income.** The weights of $\bar M_{0,0}$ are $z_{i0}$ for labor income and $a_{i,-1}/A_{ss}$ for dividends:
$$\bar M_{0,0}\,dY_0=\mathbb{E}\Big[\Big(\tfrac{1}{\mu}z_{i0}+\big(1-\tfrac{1}{\mu}\big)\tfrac{a_{i,-1}}{A_{ss}}\Big)MPC_i\Big]dY_0=\mathbb{E}\Big[\tfrac{Y_i}{Y}MPC_i\Big]dY_0$$

**Step 2: earnings heterogeneity.** The weights of $\tilde{\boldsymbol{M}}-\boldsymbol{M}$ are $Z_{ss}\,\partial z_{i0}/\partial Z_0$, with mean zero, and $\frac{1}{\mu}dY_0=dZ_0$:
$$\tfrac{1}{\mu}(\tilde M_{0,0}-M_{0,0})\,dY_0=\mathbb{E}\big[MPC_i\,Z_{ss}\,dz_{i0}\big]=\text{cov}\big(MPC_i,\,Z_{ss}\,dz_{i0}\big)$$

**Step 3: Fisher.** By section 1, $m_0-m^\chi_0=\text{cov}\big(MPC_i,\frac{a_{i,-1}}{A_{ss}}\big)$, and $dr^b_0=-(1+r_{ss})d\pi_0$:
$$B_{ss}\big(m_0-m^\chi_0\big)dr^b_0=-\text{cov}\Big(MPC_i,\,(1+r_{ss})\tfrac{B_{ss}}{A_{ss}}a_{i,-1}\Big)d\pi_0=-\text{cov}(MPC_i,NNP_i)\,d\pi_0$$

**Step 4: the rate.** A higher $r_0$ pays household $i$ an extra $a_{i0}\,dr_0$ at $t=1$, which by Fact 1 it spends as $MPC_i\,a_{i0}\frac{dr_0}{1+r_{ss}}$ today.
This holds for every household, since constrained households have $a_{i0}=0$.
Adding Fact 2,
$$M^r_{0,0}\,dr_0=\Big(\underbrace{\mathbb{E}\big[MPC_i\,a_{i0}\big]}_{\text{income effect}}-\underbrace{\tfrac{1}{\sigma}\,\mathbb{E}\big[(1-MPC_i)\,c_{i0}\big]}_{\text{substitution}}\Big)\frac{dr_0}{1+r_{ss}}$$
The capital gain and the tax cut are
$$
\begin{aligned}
-\tfrac{p}{1+r_{ss}}m_0\,dr_0&=-\mathbb{E}\Big[MPC_i\,\tfrac{p}{A_{ss}}a_{i,-1}\Big]\frac{dr_0}{1+r_{ss}}\\
-\tfrac{B_{ss}}{1+r_{ss}}M^\chi_{0,0}\,dr_0&=-\mathbb{E}[MPC_i]\,B_{ss}\frac{dr_0}{1+r_{ss}}
\end{aligned}
$$
Adding them to the income effect, with $\mathbb{E}[URE_i]=A_{ss}-p=B_{ss}$, gives
$$\mathbb{E}\big[MPC_i\,(URE_i-B_{ss})\big]\frac{dr_0}{1+r_{ss}}=\text{cov}(MPC_i,URE_i)\frac{dr_0}{1+r_{ss}}$$

Adding Steps 1 to 4 gives Auclert's equation (19).


# Appendix: proofs {-}

Household $i$ has cash on hand $x_{i0}$, consumes $c_{i0}$ and saves $a_{i0}=x_{i0}-c_{i0}\ge0$.
Let $W_i(a_{i0})$ be its expected value from $t=1$ on, so that $W_i'(a_{i0})=\beta(1+r_{ss})\mathbb{E}_0[u'(c_{i1})]$.
If the household is unconstrained, $u'(c_{i0})=W_i'(a_{i0})$.
If it is constrained, $c_{i0}=x_{i0}$ and $MPC_i=1$.

For an unconstrained household, an extra $dx$ today gives $u''\,dc_{i0}=W_i''\,(dx-dc_{i0})$, hence
$$MPC_i=\frac{W_i''}{u''+W_i''},\qquad u''+W_i''=\frac{u''}{1-MPC_i}$$

*Fact 1.* An extra $dy$ at $t=1$ raises wealth at $t=1$ from $(1+r_{ss})a_{i0}$ to $(1+r_{ss})\big(a_{i0}+\frac{dy}{1+r_{ss}}\big)$.
For an unconstrained household, this is the same as receiving $\frac{dy}{1+r_{ss}}$ today, so $dc_{i0}=MPC_i\frac{dy}{1+r_{ss}}$.
A constrained household stays at $a_{i0}=0$, so $dc_{i0}=0$.

*Fact 2.* Raise $r_0$ in the Euler equation only, $u'(c_{i0})=\frac{1+r_0}{1+r_{ss}}W_i'(a_{i0})$, at unchanged cash on hand, so $da_{i0}=-dc_{i0}$:
$$u''\,dc_{i0}=-W_i''\,dc_{i0}+u'\frac{dr_0}{1+r_{ss}}$$
Using $u''+W_i''=\frac{u''}{1-MPC_i}$ and $u'/u''=-c/\sigma$ with CRRA utility,
$$dc_{i0}=(1-MPC_i)\frac{u'}{u''}\frac{dr_0}{1+r_{ss}}=-\frac{1}{\sigma}(1-MPC_i)\,c_{i0}\frac{dr_0}{1+r_{ss}}$$
A constrained household does not move, and $1-MPC_i=0$, so the formula holds for every household.
