# Fiscal Policy in HANK: outline

Follows the slides in `tex/HANK-Fiscal.tex`, one heading per slide.
Message of the deck: the macro blocks never change, only the household block and hence $\boldsymbol{M}$.

Choices:

- TFP and the $\beta$ shock are removed, so $\pi_t=\pi_t^w$ and the only shocks are monetary and fiscal.
- The ZLB exercise is dropped.
- The Keynesian mechanism of IS-LM motivates two ingredients: nominal rigidities and a departure from Ricardian equivalence.
- RANK, TANK and HANK are different models of $\boldsymbol{M}$.
- Liquidity, the Generalized IKC and unions are extensions after HANK.
- The NK framework is the general model: markups, equity, nominal bonds, a Taylor rule, with households left open. The government chooses the debt path.
- The IKC assumes a real rate rule, real bonds and no markups. The Generalized IKC and the canonical HANK model drop these.
- All models are annual. The flexible wage slides moved to the monetary lectures, see `11-12-HANK-Monetary/flexible-wage-frames.tex`.
- Lecture 9 covers everything through TANK and ends with the exercise, lecture 10 is HANK and the extensions.

---

# Lecture 9

## Introduction

### Introduction
- Previously:
  - Economic content: Long run trends and outcomes
  - Methods: Stationary eq., transition paths and perfect foresight
- Today:
  - One New Keynesian framework
  - Three household blocks: RANK, TANK, HANK
  - Fiscal policy through the Intertemporal Keynesian Cross (IKC)
- Main message: the macro blocks never change. Only the household block changes, and it is summarized by one matrix $\boldsymbol{M}$
- Literature:
  - Auclert, Rognlie and Straub (2024), "The Intertemporal Keynesian Cross"
  - Gali textbook ch. 3-4 for background on the NK model
  - Documentation for GEModelTools

### Business cycles
- Macro variables relatively volatile around long-run trends
- Rest of the course:
  - Study how aggregate shocks cause business cycles
    - Does the transmission change with heterogeneous agents?
  - Implications for fiscal and monetary policy
- First point on agenda: Need role for monetary policy
  - Models so far in the course have featured monetary neutrality
  - Monetary policy cannot affect real quantities (unemployment, GDP)
- Figure: `FREDBC.png`

## IS-LM

### The IS-LM model
- Aggregate demand is $D=C+G$ (abstracting away from investment)
- Assume $C=c_0+c_1(Y-T)-c_2r$ with $c_1\in(0,1)$
- If output is demand determined, then $Y=D$, and $Y = \frac{1}{1-c_1}(c_0 - c_1 T + G - c_2 r)$
- Core intuition: all of income is somebody else's spending!

### Block representation
- Consumption is the only behavioral block
- Output feeds back into consumption through income
- We keep this structure for the whole lecture
- Figure: IS-LM block diagram, policy into consumption and goods market, with income feedback

### Policies in IS-LM
- Fiscal policy: multiplier is $\frac{dY}{dG}=\frac{1}{1-c_1}>1$
- Monetary policy: multiplier is $\frac{dY}{dr}=-\frac{c_2}{1-c_1}<0$
  - Direct effect is $-c_2$
  - Indirect effect through $Y$ is $c_1 \frac{dY}{dr}$
- What's missing?
  - A supply side, micro-foundations, expectations, etc.

### Can we get this in a micro-founded model?
- IS-LM: $G$ raises demand, output rises, income rises, demand rises further: a multiplier above 1
- But no budget constraints and no expectations
- In HANK we can get this back, with two ingredients:
  - Nominal rigidities: output is demand determined
  - Departure from Ricardian equivalence: the timing of income moves spending

## NK framework

### New Keynesian framework
- A proper model of business cycles and stabilization policy requires that output is partially demand determined
- The study of monetary policy requires monetary non-neutrality
- The New Keynesian (NK) model addresses these two concerns by adding to the standard model:
  - Monopolistic competition (wage-setting by unions)
  - Wage rigidities
- Textbooks use sticky prices. We use sticky wages, and the reason comes at the end of the lecture

### The model
- We write one NK model, general enough for the whole lecture
- The model consists of the following agents:
  - Households who consume and save
  - Firms with market power who produce using labor and set prices
  - Unions with market power who set wages subject to nominal rigidities
  - A mutual fund which holds stocks and bonds
  - A government that spends, taxes and issues nominal bonds
  - A central bank which follows a Taylor rule
- We leave the household problem open for now

### Overview
- Households: choose consumption and saving, work as much as the union decides
- Firms: produce with labor, set prices at a markup, pay dividends
- Unions: supply differentiated labor, set wages, face a cost of changing wages
- Mutual fund: collects savings, holds stocks and government bonds
- Government: sets spending and debt, taxes adjust
- Central bank: sets the nominal interest rate

### Block representation
- The red box is left open: RANK, TANK and HANK fill it in, nothing else changes
- $Z_{t}$ is post-tax labor income
- Figure: block diagram, household box: Households

### Firms
- Production: $Y_{t}=L_{t}$
- Prices: firms set a constant markup $\mu$ over marginal cost, prices are flexible: $P_{t}=\mu W_{t}\quad\Rightarrow\quad w_{t}\equiv\frac{W_{t}}{P_{t}}=\frac{1}{\mu}$
- So price inflation equals wage inflation: $\pi_{t}=\pi_{t}^{w}$
- Dividends, taxed at the same rate as labor income: $d_{t}=(1-\tau_{t})\left(Y_{t}-\frac{L_{t}}{\mu}\right)=(1-\tau_{t})\left(1-\frac{1}{\mu}\right)Y_{t}$

### Unions
- Unions indexed by $j\in[0,1]$ each supply a differentiated labor type (Erceg, Henderson and Levin, 2000)
- Firms use a CES bundle of labor types with sub. elasticity $\epsilon^{W}$, so labor demand for union $j$ is $L_{t}^{j}=\left(\frac{W_{t}^{j}}{W_{t}}\right)^{-\epsilon^{W}}L_{t}$
- Union $j$ sets its wage $W_{t}^{j}$ and supplies what firms demand
- Changing wages has a quadratic utility cost $\Rightarrow$ sticky wages
- Note: no resource cost, so goods market clearing is unchanged

### Union problem
- Bellman equation of the union: the state is last period's wage subject to labor demand
- FOC wrt. $W_{t}^{j}$, using labor demand:
- Envelope condition: $V_{t}^{\prime}\left(W_{t-1}^{j}\right)=\theta^{W}\left(\frac{W_{t}^{j}}{W_{t-1}^{j}}-1\right)\frac{W_{t}^{j}}{\left(W_{t-1}^{j}\right)^{2}}$

### Derivation of W-NKPC
- Symmetry: all unions set the same wage, $W_{t}^{j}=W_{t}$, $L_{t}^{j}=L_{t}$
- Multiply the FOC by $W_{t}/\theta^{W}$, with $W_{t}/P_{t}=1/\mu$: $\pi_{t}^{w}\left(1+\pi_{t}^{w}\right)=\kappa\left\{ h'\left(L_{t}\right)-\frac{1-\tau_{t}}{\mu\mu^{w}}u'\left(C_{t}\right)\right\} L_{t}+\beta E_{t}\pi_{t+1}^{w}\left(1+\pi_{t+1}^{w}\right)$ with slope $\kappa=\frac{\epsilon^{W}}{\theta^{W}}$ and wage markup $\mu^{w}=\frac{\epsilon^{W}}{\epsilon^{W}-1}$
- Simpler version with the same first-order approximation around $\pi^{w}=0$, $L=1$: $\pi_{t}^{w}=\kappa\left(h'\left(L_{t}\right)-\frac{1-\tau_{t}}{\mu\mu^{w}}u'\left(C_{t}\right)\right)+\beta E_{t}\pi_{t+1}^{w}$

### What sticky wages mean
- Nominal wages $W_{t}$ adjust only partly, slowly through the W-NKPC
- At this wage, unions supply whatever labor firms ask for
- Firms produce what is demanded and hire $L_{t}=Y_{t}$
- So output is demand determined: demand sets $Y_{t}$, and $Y_{t}$ sets $L_{t}$
- Households are off their labor supply curve in the short run
- This is possible because $\mu^{w}>1$: the wage is above the marginal rate of substitution, so working a bit more is still worth it
- Over time, unions raise wages if $L_{t}$ is high: that is the W-NKPC

### Fiscal authority
- Budget: spending $G_{t}$, taxes $T_{t}$, nominal debt with real value $B_{t}$ $T_{t}=(1+r_{t}^{b})B_{t-1}+G_{t}-B_{t},\quad T_{t}=\tau_{t}Y_{t},\quad1+r_{t}^{b}=\frac{1+i_{t-1}}{1+\pi_{t}}$
- The government chooses the paths of $G_{t}$ and $B_{t}$, with $B_{t}\rightarrow B_{ss}$
- Taxes adjust to satisfy the budget
- Deficit: with a constant return on debt, and $\boldsymbol{L}$ the lag matrix, it only depends on the debt path: $d\boldsymbol{G}-d\boldsymbol{T}=\left(\boldsymbol{I}-(1+r_{ss})\boldsymbol{L}\right)d\boldsymbol{B}$
- Debt returns to steady state, so with $q_{t}=(1+r_{ss})^{-t}$ the deficit has zero present value: $\boldsymbol{q}^{\prime}(d\boldsymbol{G}-d\boldsymbol{T})=0$
- Balanced budget: $B_{t}=B_{ss}$
- Deficit financing: debt rises first, and taxes repay it later

### Central bank
- Monetary policy follows a Taylor rule: $i_{t}=i_{ss}+\phi\pi_{t}$
- Fisher relationship links the nominal and the real rate: $1+r_{t}=\frac{1+i_{t}}{1+\pi_{t+1}}$
- Monetary policy matters because wages are sticky: inflation does not move 1-1 with $i_{t}$, so $r_{t}$ moves

### Mutual fund
- Collects household savings and holds stocks and bonds: $A_{t}=p_{t}+B_{t}$
- A mass 1 of shares with price $p_{t}$. No arbitrage with bonds:
- Return from $t=1$ on: known in advance, so $r_{t+1}^{p}=r_{t}$
- Return at $t=0$: a surprise shock revalues the portfolio $1+r_{0}^{p}=\underset{\text{capital gains}}{\underbrace{\frac{p_{0}+d_{0}}{A_{ss}}}}+\underset{\text{inflation revaluation}}{\underbrace{\frac{1+i_{ss}}{1+\pi_{0}}\frac{B_{ss}}{A_{ss}}}}$
- Inflation: bonds pay the nominal rate $i_{ss}$ set before the shock. If $\pi_{0}$ rises, their real value falls, and so does $r_{0}^{p}$
- Stocks: if expected dividends rise or $r$ falls, $p_{0}$ rises, and so does $r_{0}^{p}$

### Households
- Households save in the mutual fund and take income and returns as given: $C_{t}+A_{t}=\left(1+r_{t}^{p}\right)A_{t-1}+Z_{t},\quad Z_{t}=\frac{1-\tau_{t}}{\mu}L_{t}^{hh}$
- Labor supply $L_{t}^{hh}$ is set by the union
- We leave the household problem open for now. All we need is an aggregate consumption function: $\boldsymbol{C}^{hh}=C^{hh}\left(\boldsymbol{Z},\boldsymbol{r}^{p}\right)$
- Consumption depends on the whole path of income and of $r_{t}^{p}$, the return on the mutual fund. Households do not care about $r_{t}$ or $i_{t}$ directly

### Equilibrium
- Three markets: $Y_{t}=C_{t}+G_{t},\quad L_{t}^{hh}=L_{t},\quad A_{t}=p_{t}+B_{t}$
- By Walras' law, we can drop one of them: the diagram targets goods, the HANK code targets assets
- Goods market has the same structure as IS-LM: $\boldsymbol{Y}=\boldsymbol{G}+C^{hh}(\boldsymbol{Z},\boldsymbol{r}^{p})$
- Supply side: the W-NKPC, with feedback through the Taylor rule
- In general, solving this requires a computer

### Equations summary
- Households: $\boldsymbol{C}=C^{hh}(\boldsymbol{Z},\boldsymbol{r}^{p})$, $C_{t}+A_{t}=(1+r_{t}^{p})A_{t-1}+Z_{t}$, $Z_{t}=\frac{1-\tau_{t}}{\mu}L_{t}$
- Firms: $Y_{t}=L_{t}$, $\pi_{t}=\pi_{t}^{w}$, $d_{t}=(1-\tau_{t})(1-1/\mu)Y_{t}$
- Union: $\pi_{t}^{w}=\kappa\left(h'(L_{t})-\frac{1-\tau_{t}}{\mu\mu^{w}}u'(C_{t})\right)+\beta\pi_{t+1}^{w}$
- Mutual fund: $A_{t}=p_{t}+B_{t}$, $1+r_{t}=\frac{p_{t+1}+d_{t+1}}{p_{t}}=\frac{1+i_{t}}{1+\pi_{t+1}}$, $r_{t+1}^{p}=r_{t}$
- Government: $T_{t}=(1+r_{t}^{b})B_{t-1}+G_{t}-B_{t}$, $T_{t}=\tau_{t}Y_{t}$
- Central bank: $i_{t}=i_{ss}+\phi\pi_{t}$
- Market clearing: $Y_{t}=C_{t}+G_{t}$
- Exogenous: $G_{t}$ and $B_{t}$
- $C^{hh}$ is the only equation that changes across RANK, TANK and HANK

## Deriving the IKC for a fiscal shock

### Three assumptions
- To get analytical results, we make three assumptions
  - Real rate rule: $1+i_{t}=(1+r_{ss})(1+\pi_{t+1})$, so $r_{t}=r_{ss}$
    - A Taylor rule with a coefficient of 1 on expected inflation
    - A middle ground between loose policy (ZLB) and tight policy (active Taylor rule)
  - Real bonds: no revaluation of debt from surprise inflation at $t=0$
  - No markups: $\mu=1$, so $d_{t}=0$, $p_{t}=0$ and $A_{t}=B_{t}$
- Together: $r_{t}^{p}=r_{ss}$ at all dates, so $C^{hh}(\boldsymbol{Z},\boldsymbol{r}^{p})$ becomes $C^{hh}(\boldsymbol{Z})$
- Inflation is irrelevant to output

### Simplified system of equations
- \small
- The debt path and the government budget give $\boldsymbol{T}$
- Goods market clearing alone then gives $\boldsymbol{Y}$: $\boldsymbol{Y}=\boldsymbol{G}+C^{hh}(\boldsymbol{Y}-\boldsymbol{T})$
- The W-NKPC still gives inflation, and the central bank gives $i_{t}$, but not required to determine $Y$.

### Block representation of the IKC model
- \ikcdag{Households}{$C^{hh}(\boldsymbol{Z})$}
- One unknown, one target: $r$ is fixed, and firms and the fund play no role
- Taxes depend only on the shocks, not on $\boldsymbol{Y}$

### Why $C(\boldsymbol{Z})$ and not $C(\boldsymbol{Y},\boldsymbol{T})$?
- Income: with $\mu=1$, $d_{t}=0$, so all of $Y_{t}$ is labor income
- Taxes: $T_{t}=\tau_{t}Y_{t}$ hit that same income in proportion: $Z_{t}=(1-\tau_{t})Y_{t}=Y_{t}-T_{t}$
- Households only see the difference: one more unit of output and one less unit of taxes have the same effect
- So one matrix is enough, and it applies to $d\boldsymbol{Y}-d\boldsymbol{T}$
- Trivial in RANK, an assumption in TANK and HANK: every household's income must move in proportion, $z_{it}Z_{t}$
- Without it (lump-sum taxes in HANK, profits paid to the rich), $\boldsymbol{Y}$ and $\boldsymbol{T}$ enter separately and the balanced budget result below fails

### Aggregate consumption function in the sequence space
- Households are forward-looking, and the whole path of income is known at $t=0$
- Consumption at date $t$ depends on income at all dates: $C_{t}^{hh}=C_{t}^{hh}(Z_{0},Z_{1},Z_{2},\dots)$ for all $t$. Stacking gives $\boldsymbol{C}^{hh}=C^{hh}(\boldsymbol{Z})$
- Total differentiation: $dC_{t}^{hh}=\sum_{s=0}^{\infty}\frac{\partial C_{t}^{hh}}{\partial Z_{s}}dZ_{s}$
- Vector form: $d\boldsymbol{C}^{hh}=\boldsymbol{M}d\boldsymbol{Z}$, with $M_{t,s}=\frac{\partial C_{t}^{hh}}{\partial Z_{s}}$ the matrix of iMPCs
- $M_{t,s}$: how much consumption at date $t$ moves when income at date $s$ moves
- $\boldsymbol{M}$ encodes the entire complexity of HH behavior

### The iMPC matrix $\boldsymbol{M}$
- Matrix with entries $M_{t,s}$, column $s=1$ highlighted in red, row $t=2$ in blue
- Column $s$: the path of consumption after one more unit of income at date $s$. Each column is an impulse response
- For $s>0$, households learn at $t=0$ that income rises at $s$, so they can react before it arrives
- Row $t$: how much consumption at date $t$ reacts to income at each date
- $M_{0,0}$ is essentially the annual MPC

### Intertemporal Keynesian Cross
- Combine goods market clearing and the consumption function: $\boldsymbol{Y}=\boldsymbol{G}+C^{hh}(\boldsymbol{Y}-\boldsymbol{T})$
- Differentiate, using $d\boldsymbol{C}^{hh}=\boldsymbol{M}d\boldsymbol{Z}$ and $d\boldsymbol{Z}=d\boldsymbol{Y}-d\boldsymbol{T}$: $d\boldsymbol{Y}=d\boldsymbol{G}+\boldsymbol{M}(d\boldsymbol{Y}-d\boldsymbol{T})\Leftrightarrow(\boldsymbol{I}-\boldsymbol{M})d\boldsymbol{Y}=d\boldsymbol{G}-\boldsymbol{M}d\boldsymbol{T}$
- The whole model reduces to one equation, and $\boldsymbol{M}$ is a sufficient statistic: same iMPCs imply the same $d\boldsymbol{Y}$

### Balanced budget multiplier
- Balanced budget: debt does not move, $d\boldsymbol{B}=\boldsymbol{0}$, so $d\boldsymbol{T}=d\boldsymbol{G}$
- The IKC becomes
- So $d\boldsymbol{Y}=d\boldsymbol{G}$ solves it, for any $\boldsymbol{M}$
- Intuition: output rises by $dG_{t}$ and taxes by $dG_{t}$, so post-tax income does not change, and neither does consumption: $d\boldsymbol{C}=\boldsymbol{0}$
- Multiplier of exactly 1, whatever the household block: RANK, TANK or HANK
- So the household block only matters for deficits

### Illustration
- Writing out the IKC with full vectors and the matrix $\boldsymbol{M}$

### Tricky point: inverting $\boldsymbol{I}-\boldsymbol{M}$
- Issue: we cannot solve the IKC by inverting $\boldsymbol{I}-\boldsymbol{M}$. In every model with a budget constraint, households eventually spend every unit of income, so $\boldsymbol{q}^{\prime}(\boldsymbol{I}-\boldsymbol{M})=\boldsymbol{0}\quad\text{(proof on the next slide)}$
- One equation is redundant: multiply the IKC by $\boldsymbol{q}^{\prime}$ This is just the government budget: it holds for any $d\boldsymbol{Y}$
- Many solutions: in RANK, $\boldsymbol{M}^{RA}\boldsymbol{1}=\boldsymbol{1}$, so a permanent rise in $d\boldsymbol{Y}$ can be added to any solution
- We pick the one where $d\boldsymbol{Y}$ returns to zero: $d\boldsymbol{Y}=\mathcal{M}(d\boldsymbol{G}-\boldsymbol{M}d\boldsymbol{T}),\quad\mathcal{M}:\text{ inverse of }\boldsymbol{I}-\boldsymbol{M}\text{ on paths that return to zero}$
- Numerically: truncation at a finite horizon makes $\boldsymbol{I}-\boldsymbol{M}$ invertible, but precision worsens with the horizon

### Tricky point: proof
- Step 1: linearize the household budget, with $dA_{-1}=0$ and constant $r$: $dC_{t}+dA_{t}=(1+r_{ss})dA_{t-1}+dZ_{t}$
- Step 2: multiply by $q_{t}=(1+r_{ss})^{-t}$ and sum over $t$. Since $q_{t}(1+r_{ss})=q_{t-1}$, the asset terms cancel out: $\sum_{t=0}^{T}q_{t}dC_{t}+q_{T}dA_{T}=\sum_{t=0}^{T}q_{t}dZ_{t}$
- Step 3: let $T\rightarrow\infty$, with assets back at steady state, $dA_{T}\rightarrow0$: $\boldsymbol{q}^{\prime}d\boldsymbol{C}^{hh}=\boldsymbol{q}^{\prime}d\boldsymbol{Z}$
- Step 4: use $d\boldsymbol{C}^{hh}=\boldsymbol{M}d\boldsymbol{Z}$. This holds for any $d\boldsymbol{Z}$, so $\boldsymbol{q}^{\prime}\boldsymbol{M}=\boldsymbol{q}^{\prime}\quad\Leftrightarrow\quad\boldsymbol{q}^{\prime}(\boldsymbol{I}-\boldsymbol{M})=\boldsymbol{0}$
- So no inverse: if $(\boldsymbol{I}-\boldsymbol{M})^{-1}$ existed, multiplying by it would give $\boldsymbol{q}^{\prime}=\boldsymbol{0}$

### Solving the IKC
- Rewrite the IKC in terms of consumption:
- $d\boldsymbol{T}$ is exogenous: $dT_{t}=dG_{t}+(1+r_{ss})dB_{t-1}-dB_{t}$. The tax rate $\tau_{t}=T_{t}/Y_{t}$ adjusts
- Deficits: $d\boldsymbol{G}\neq d\boldsymbol{T}$
  - All the action in $d\boldsymbol{C}$ comes from the deficit
  - This is where the household block matters

### Fiscal multipliers
- Impact multiplier: $\frac{\partial Y_{0}}{\partial G_{0}}$ Cumulative multiplier: $\frac{\sum_{t=0}^{\infty}(1+r_{ss})^{-t}dY_{t}}{\sum_{t=0}^{\infty}(1+r_{ss})^{-t}dG_{t}}=\frac{\boldsymbol{q}^{\prime}d\boldsymbol{Y}}{\boldsymbol{q}^{\prime}d\boldsymbol{G}}$

### What do we know about $\boldsymbol{M}$?
- If we knew all of $\boldsymbol{M}$ from the data, we would not need a model: the IKC gives $d\boldsymbol{Y}$
- We only observe the first column: Fagereng et al., lottery winnings
- Annual MPC above 0.5, then 0.18 the year after
- Other columns (announcements) are very hard to estimate. So we need models: RANK, TANK and HANK are different microfoundations of $\boldsymbol{M}$
- Figure: `iMPC_data.png`

### Rest of the course: different models of $\boldsymbol{M}$
- RANK, TANK and HANK are different microfoundations of $C^{hh}$, and so of $\boldsymbol{M}$
- Spoiler for the rest of the class:
- Table: \| Ricardian equivalence \| Impact \| Cumulative
- Table: RANK \| yes \| $1$ \| $1$
- Table: TANK \| no \| $>1$ \| $1$
- Table: HANK \| no \| $>1$ \| $>1$

## RANK

### Block representation
- \ikcdag{Rep. household}{$C^{hh}(\boldsymbol{Z})$}
- RANK fills the red box with one representative household

### Household problem in RANK
- The representative household solves: $\max_{C_{t},A_{t}} \sum_{t=0}^{\infty}\beta^{t}\left[u\left(C_{t}\right)-h\left(L_{t}^{hh}\right)\right] ,\quad \text{s.t. }C_{t}+A_{t} =\left(1+r_{ss}\right)A_{t-1}+Z_{t}$
- Labor supply $L_{t}^{hh}$ is set by the union, so only one first-order condition: $u'\left(C_{t}\right) =\beta\left(1+r_{ss}\right)u'\left(C_{t+1}\right)$
- No borrowing constraint, no idiosyncratic risk

### Consumption function in RANK
- Step 1: in steady state, Euler requires $\beta(1+r_{ss})=1$. With $r_{t}=r_{ss}$, it becomes $u'(C_{t})=u'(C_{t+1})$, so $C_{t+s}=C_{t}$ for all $s$
- Step 2: iterate the budget forward, with $(1+r_{ss})^{-1}=\beta$: $\sum_{s=0}^{\infty}\beta^{s}C_{t+s}=(1+r_{ss})A_{t-1}+\sum_{s=0}^{\infty}\beta^{s}Z_{t+s}$
- Step 3: use constant consumption, $\sum_{s}\beta^{s}C_{t+s}=\frac{C_{t}}{1-\beta}$, and $(1-\beta)(1+r_{ss})=r_{ss}$: $C_{t}=(1-\beta)\sum_{s=0}^{\infty}\beta^{s}Z_{t+s}+r_{ss}A_{t-1}$
- Households consume their permanent income: only the present value of income matters

### iMPC in RANK
- News arrives at date 0, so evaluate Step 3 at $t=0$ with $dA_{-1}=0$: one unit of income at date $s$ raises $C_{0}$ by $(1-\beta)q_{s}$, with $q_{s}=\beta^{s}$
- Consumption is constant, so it rises by the same amount at all dates, before and after $s$. At later dates this runs through $A_{t-1}$: $M_{t,s}=(1-\beta)\beta^{s}\quad\Rightarrow\quad\boldsymbol{M}^{RA}=(1-\beta)\boldsymbol{1}\boldsymbol{q}^{\prime}$
- Flat and very low, about $0.02$ per year: far from the data
- Figure: `iMPC_RANK.pdf`

### Ricardian equivalence in RANK
- Recall that the deficit has zero present value for any $d\boldsymbol{G},d\boldsymbol{B}$: $\boldsymbol{q}^{\prime}(d\boldsymbol{G}-d\boldsymbol{T})=0$
- And it disappears in RANK: $\boldsymbol{M}^{RA}(d\boldsymbol{G}-d\boldsymbol{T})=(1-\beta)\boldsymbol{1}\boldsymbol{q}^{\prime}(d\boldsymbol{G}-d\boldsymbol{T})=\boldsymbol{0}$
- Hence $d\boldsymbol{C}=\boldsymbol{0}$ and $d\boldsymbol{Y}=d\boldsymbol{G}$ for any $d\boldsymbol{G},d\boldsymbol{B}$
- Impact and cumulative multipliers are exactly 1, whatever the financing

### Ricardian equivalence: intuition
- The representative household smooths consumption perfectly
- It only cares about permanent income
- The government can shift income through time, with debt
- But it does not change permanent income: a tax cut today is a tax rise later
- So consumption does not change

### Is RANK realistic?
- MPCs are too low: $0.02$ per year against about $0.5$ in the data
- Households are too forward-looking: income far in the future counts at its full present value
- No borrowing constraint: every household can smooth

## TANK

### Household problem in TANK
- Share $\lambda$: hand-to-mouth. No savings, consume all income, $MPC=1$: $C_{t}^{H}=Z_{t}$
- Share $1-\lambda$: Ricardian. Same problem as in RANK, hold all the bonds
- Same labor and same tax rate for both types, so $Z_{t}=Y_{t}-T_{t}$ for both
- Aggregate consumption: $C_{t}^{hh}=\lambda Z_{t}+(1-\lambda)C_{t}^{R}$

### Block representation
- \ikcdag{Two types}{$C^{hh}(\boldsymbol{Z})$}
- Same picture as RANK: only the content of the red box changes

### iMPC in TANK
- The $\boldsymbol{M}$ matrix is a weighted sum: $\boldsymbol{M}^{TA}=(1-\lambda)\boldsymbol{M}^{RA}+\lambda\boldsymbol{I}$
- $C_{t}$ overweights income at date $t$, but only at that same date
- Here $\lambda$ is set to match the annual MPC of baseline HANK
- Figure: `iMPC_TANK.pdf`

### Solving TANK
- Intertemporal Keynesian Cross with $\boldsymbol{M}^{TA}$:
- This is the RANK IKC with a modified spending shock. Solution: $d\boldsymbol{Y}=d\boldsymbol{G}+\frac{\lambda}{1-\lambda}\left[d\boldsymbol{G}-d\boldsymbol{T}\right]$
- Same form as the IS-LM Keynesian cross, with $\lambda$ in the role of $c_{1}$
- Amplification with deficit financing, increasing in $\lambda$

### TANK Proof
- In TANK we have:
- where $d\tilde{\boldsymbol{G}}=\frac{1}{1-\lambda}\left[d\boldsymbol{G}-\lambda d\boldsymbol{T}\right]$
- Recall that in RANK $d\boldsymbol{Y}=d\boldsymbol{G}+\boldsymbol{M}^{RA}(d\boldsymbol{Y}-d\boldsymbol{T})$ with solution $d\boldsymbol{Y}=d\boldsymbol{G}$ thus implying $d\boldsymbol{Y}=d\tilde{\boldsymbol{G}}$ in TANK:

### Impact and cumulative multipliers in TANK
- Impact multiplier if debt absorbs all of the spending at $t=0$, so $dT_{0}=0$: $\frac{dY_{0}}{dG_{0}}=\frac{1}{1-\lambda}>1$
- Cumulative multiplier is still one:
- The boom on impact is exactly offset by the bust when taxes are repaid

### What TANK misses
- $dC_{t}$ depends only on the deficit at the same date
  - No persistence: the boost ends when the deficit ends
  - Consumption falls during repayment
- No intertemporal MPCs: it can match the MPC on impact, but $M_{1,0}$ is far too low
- Stylized level of inequality
- Hard to connect to micro data
- No precautionary saving

### Exercise: RANK and TANK
- Familiarize yourself with the model equations in blocks.py. Do you understand all the equations?
- Compute IRFs to a monetary policy shock and to a government spending shock under the Taylor rule. Use model.find_IRFs().
- Derive the IKC in RANK under a constant real rate. Compute $\boldsymbol{M}^{RA}$, solve for $d\boldsymbol{Y}$, and compare with model.find_IRFs().
- Compute the impact and cumulative multipliers. How do they depend on the financing of the spending?
- Do the same with TANK, using $\boldsymbol{M}^{TA}$. How do the multipliers change with $\lambda$?

---

# Lecture 10

## HANK

### Household problem in HANK
- Household problem:
- Active decisions: Consumption-saving, $c_{t}$ (and $a_{t}$)
- Union decision: Labor supply, $\ell_{t}$
- Aggregate Consumption: $C_{t}^{hh}=\int c_{t}d\mathcal{D}_{t}$
- Consumption function: $\boldsymbol{C}^{hh}=C^{hh}(\boldsymbol{Z},\boldsymbol{r})$ with $Z_{t}=\left(1-\tau_{t}\right)\ell_{t}$ and transfers $\chi_{t}=0$ at baseline

### Block representation
- \ikcdag{Heterogeneous}{$C^{hh}(\boldsymbol{Z})$}
- $C^{hh}(\boldsymbol{Z})$ is unchanged as an object
- We now compute it from the distribution of households

### iMPC in HANK
- $\boldsymbol{M}$ is computed numerically: the household Jacobian with respect to $Z$, as in lectures 5-6
- Spending is spread around the date of the income: anticipated before, gradual after
- Past and future income matter at all horizons, not only the same date
- Figure: `iMPC_compare.pdf`

### iMPCs in the data: yes!
- Same shape: high on impact, then a gradual decline
- Baseline model: 0.36 and 0.15. Data: 0.51 and 0.18
- RANK has no decline, TANK has a drop after one period
- We only observe the first column, the rest comes from the model
- Figure: `iMPC_data.png`
- Figure: `iMPC_annual.pdf`

### Consumption function in HANK
- Present value weights $M_{3,s}(1+r)^{s}$ peak at date 3. They are flat in RANK
- Two properties:
  - Front-loaded: income is spent soon after it arrives
  - Discounted: households react less to future changes than the present value implies
- Figure: `M_row.pdf`

### Why this breaks Ricardian equivalence
- With $D_{s}=dG_{s}-dT_{s}$, the first round of spending is $\sum_{s}M_{t,s}D_{s}=\sum_{s}M_{t,s}(1+r_{ss})^{s}q_{s}D_{s}$
- Flat weights (RANK): the sum is proportional to $\boldsymbol{q}^{\prime}\boldsymbol{D}=0$, so deficits are neutral
- Peaked weights (HANK): larger weights on the early deficit than on the later repayment give a boost
- So the timing of income matters, not just its present value

### Rounds of spending
- My income is your spending, spread over time: $d\boldsymbol{C}=\sum_{k\geq1}\boldsymbol{M}^{k}(d\boldsymbol{G}-d\boldsymbol{T})$
- The first round explains most of the impact
- Later rounds explain the persistence
- Figure: `rounds.pdf`

### Multipliers and debt persistence
- Debt path: $dB_{t}=\rho_{B}\left(dB_{t-1}+dG_{t}\right)$, and taxes follow from the budget
- More persistent debt means larger multipliers, on impact and cumulatively
- Aggregate estimates are 0.6-0.8, and the data does not reject up to 1.5 (Ramey, 2019)
- Figure: `multipliers_rho_B.pdf`

### Why is the cumulative multiplier above 1?
- TANK: Ricardian households hold any amount of debt at $r_{ss}$. Debt needs no boom, so $\boldsymbol{q}^{\prime}d\boldsymbol{C}=0$
- HANK: households must want to hold the extra debt. At fixed $r$, only higher income makes them want more assets
- So persistent debt needs a persistent boom
  - Baseline with $\rho_{B}=0.9$: long after $dG_{t}$ is gone, $dY_{t}\approx0.47\,dB_{t}$
  - More persistent debt lasts longer, so the boom lasts longer

### Takeaway
- Multipliers on deficit-financed $G$, at constant $r$:
- Same macro blocks: only the household block, and so $\boldsymbol{M}$, changes
- Table: \| Ricardian equivalence \| Impact \| Cumulative
- Table: RANK \| yes \| $1$ \| $1$
- Table: TANK \| no \| $>1$ \| $1$
- Table: HANK \| no \| $>1$ \| $>1$

## Self-financing deficits

### Self-financing deficits
- So far the debt path is chosen and the tax rate adjusts. Now the reverse (Angeletos et al., 2023): $\tau$ stays fixed and debt adjusts
- A rise in $G$ is financed by debt, and no tax hike ever comes: $d\boldsymbol{T}=\tau d\boldsymbol{Y}$
- IKC: $(\boldsymbol{I}-(1-\tau)\boldsymbol{M})d\boldsymbol{Y}=d\boldsymbol{G}$. With $\tau>0$ the inverse can exist
- Multiply by $\boldsymbol{q}^{\prime}$: $\tau\boldsymbol{q}^{\prime}d\boldsymbol{Y}=\boldsymbol{q}^{\prime}d\boldsymbol{G}$, so the cumulative multiplier is exactly $1/\tau$

### Self-financing deficits: illustration
- Debt rises, then returns to steady state without any tax hike
- This needs rows of $\boldsymbol{M}$ that decline. In RANK, output rises forever and debt never returns
- Holds at constant $r$. With nominal debt, inflation also erodes part of the debt
- Figure: `self_financing.pdf`

### How much of a deficit is self-financed?
- For any debt path: $\boldsymbol{q}^{\prime}d\boldsymbol{T}=\boldsymbol{q}^{\prime}d\boldsymbol{G}$
- $dT_{t}\approx\tau_{ss}dY_{t}+Y_{ss}d\tau_{t}$, so $Y_{ss}\boldsymbol{q}^{\prime}d\boldsymbol{\tau}=(1-\tau_{ss}\times\frac{\boldsymbol{q}^{\prime}d\boldsymbol{Y}}{\boldsymbol{q}^{\prime}d\boldsymbol{G}})\boldsymbol{q}^{\prime}d\boldsymbol{G}$
- Self-financed share: $\tau_{ss}\times$ cumulative multiplier
- Cumulative multiplier above $1/\tau_{ss}$: the deficit is fully self-financed, and the tax rate can even fall
- Baseline: $1/\tau_{ss}\approx4.9$. $\rho_{B}=0.75$: multiplier 1.8, 37%. $\rho_{B}=0.9$: multiplier 5.4, 110%
- Persistent debt is what makes self-financing possible

## Liquidity

### Liquidity: low liquidity
- So far the only asset is government bonds
- We can calibrate $B/Y$ to get an annual MPC of about 0.5
- This gives roughly the amount of liquid wealth in the data
- The baseline model is of this type: $B=0.25$ of annual output
- Called HA-one in Auclert et al.

### Liquidity: high wealth
- But total wealth in the data is much larger
- Assume $\mu>1$ and match $(p+B)/Y=500%$ of annual GDP
- Then MPCs are too low: households have enough assets to smooth income shocks
- Solution: heterogeneous $\beta$, in Auclert, Rognlie and Straub (2025)
  - Rich households hold most of the wealth
  - The bottom 90% have higher MPCs and little wealth

### Liquidity: two assets
- Other solution: two-asset models with high total wealth (Auclert et al.; Kaplan, Moll and Violante)
- Wealthy hand-to-mouth households: positive net worth but low liquid wealth (housing)
- Match high wealth and high MPCs
- Called HA-two in Auclert et al.

### Back to the data
- HA-one and HA-two fit the first column
- HA-hi-liq, a one-asset model at high wealth, does not. Neither does RA or TA
- Figure: `iMPC_models`

### Back to multipliers
- Models that fit the iMPCs have similar impact multipliers
- Cumulative multipliers are above 1 in both, but much lower in HA-two: savings migrate to the illiquid account, so iMPCs decline more slowly
- Figure: `multiplier`

## Generalized IKC

### Generalized IKC
- So far: real rate rule, real bonds, $\mu=1$
- We now drop these assumptions one at a time, starting each time from the IKC:
  - Markups: $\mu>1$, so $p>0$ and capital gains
  - Taylor rule: inflation moves $r$, so we need the W-NKPC
  - Nominal bonds: surprise inflation revalues debt at $t=0$
- Then we put all of them together

### 1. Markups: consumption depends on capital gains
- With $\mu>1$, income is split in two: $Z_{t}=\frac{1}{\mu}(Y_{t}-T_{t})$, $d_{t}=(1-\frac{1}{\mu})(Y_{t}-T_{t})$
- Households receive $Z_{t}$ directly. Dividends go through the mutual fund: a shock at $t=0$ changes all expected dividends, so the stock price jumps
- Capital gain: the first-period return pays this out in one go, with real bonds: $\text{cap}_{0}\equiv A_{ss}dr_{0}^{p}=dp_{0}+dd_{0}$
- Consumption: $d\boldsymbol{C}=\frac{1}{\mu}\boldsymbol{M}(d\boldsymbol{Y}-d\boldsymbol{T})+\boldsymbol{m}\text{cap}_{0}$, with $\boldsymbol{m}$ the iMPCs out of capital gains
- Goal: write $\text{cap}_{0}$ as a function of $d\boldsymbol{Y}$ and $d\boldsymbol{T}$

### 1. Markups: capital gain as a function of $d\boldsymbol{Y}$
- Step 1: no arbitrage at constant $r$: $p_{t}=\frac{p_{t+1}+d_{t+1}}{1+r_{ss}}$
- Step 2: iterate forward, no bubble. The stock price is the present value of dividends: $p_{0}+d_{0}=\sum_{t=0}^{\infty}q_{t}d_{t}=\boldsymbol{q}^{\prime}\boldsymbol{d}$
- Step 3: differentiate at constant $r$: $\text{cap}_{0}=dp_{0}+dd_{0}=\boldsymbol{q}^{\prime}d\boldsymbol{d}$
- Step 4: use $d\boldsymbol{d}=(1-\frac{1}{\mu})(d\boldsymbol{Y}-d\boldsymbol{T})$: $\text{cap}_{0}=(1-\frac{1}{\mu})\boldsymbol{q}^{\prime}(d\boldsymbol{Y}-d\boldsymbol{T})$

### 1. Markups: IKC with capital gains
- Replace $\text{cap}_{0}$ in consumption, and use $d\boldsymbol{Y}=d\boldsymbol{C}+d\boldsymbol{G}$: $d\boldsymbol{Y}=d\boldsymbol{G}+\bar{\boldsymbol{M}}(d\boldsymbol{Y}-d\boldsymbol{T})$, with $\bar{\boldsymbol{M}}\equiv\frac{1}{\mu}\boldsymbol{M}+(1-\frac{1}{\mu})\boldsymbol{m}\boldsymbol{q}^{\prime}$
- Same IKC: $\bar{\boldsymbol{M}}$ is an average of two iMPCs
- Still $\boldsymbol{q}^{\prime}\bar{\boldsymbol{M}}=\boldsymbol{q}^{\prime}$, since $\boldsymbol{q}^{\prime}\boldsymbol{m}=1$
- Everything holds with $\bar{\boldsymbol{M}}$: balanced budget multiplier of 1, RANK unchanged since $\boldsymbol{m}\boldsymbol{q}^{\prime}=\boldsymbol{M}^{RA}$
- $\boldsymbol{m}$ is small, so $\bar{\boldsymbol{M}}$ is below $\boldsymbol{M}$ on impact

### 2. Taylor rule: $r$ moves
- Back to $\mu=1$ and real bonds, with $i_{t}=i_{ss}+\phi\pi_{t}$
- Real rate: $d\boldsymbol{r}=(\phi\boldsymbol{I}-(1+r_{ss})\boldsymbol{F})d\boldsymbol{\pi}$, with $\boldsymbol{F}$ the lead matrix
- $d\boldsymbol{Y}=d\boldsymbol{G}+\boldsymbol{M}(d\boldsymbol{Y}-d\boldsymbol{T})+\boldsymbol{M}^{r}d\boldsymbol{r}$
- Taxes: $d\boldsymbol{T}=d\boldsymbol{G}-(\boldsymbol{I}-(1+r_{ss})\boldsymbol{L})d\boldsymbol{B}+B_{ss}\boldsymbol{L}d\boldsymbol{r}$
- Output now depends on inflation, so the W-NKPC is no longer a side equation

### 2. Taylor rule: two vector equations
- Linearized W-NKPC, with $Y_{ss}=1$: $d\boldsymbol{\pi}=\boldsymbol{K}(\nu d\boldsymbol{Y}+\frac{\sigma}{C_{ss}}(d\boldsymbol{Y}-d\boldsymbol{G})+\frac{d\boldsymbol{\tau}}{1-\tau_{ss}})$, $\boldsymbol{K}\equiv\kappa h'(1)(\boldsymbol{I}-\beta\boldsymbol{F})^{-1}$, $d\boldsymbol{\tau}=d\boldsymbol{T}-\tau_{ss}d\boldsymbol{Y}$
- Two vector equations in $d\boldsymbol{Y},d\boldsymbol{\pi}$: one stacked linear system of size $2T$
- Mechanism: $G$ raises $Y$, so wages and $\pi$ rise, the central bank raises $r$, and $C$ is crowded out
- With $\kappa=0$ we are back to the IKC. Higher $\kappa$ or $\phi$ means more crowding out

### 2. Taylor rule: government spending shock
- Canonical HANK model, real bonds
- $G$ raises demand, so wages and inflation rise. The Taylor rule raises $r$, which crowds out $C$
- Impact multiplier falls from 1.63 to 1.17, cumulative from 1.23 to 0.43
- Figure: `HANK_G_taylor.pdf`

### 3. Nominal bonds
- Back to $\mu=1$ and the real rate rule, with nominal bonds
- Surprise inflation lowers the return on debt at $t=0$ only: $dr_{0}^{b}\approx-(1+r_{ss})d\pi_{0}$
- Bondholders lose: $\text{cap}_{0}=B_{ss}dr_{0}^{b}$
- Taxpayers gain: $d\boldsymbol{T}=d\boldsymbol{G}-(\boldsymbol{I}-(1+r_{ss})\boldsymbol{L})d\boldsymbol{B}+\boldsymbol{e}_{0}B_{ss}dr_{0}^{b}$
- IKC: $d\boldsymbol{Y}=d\boldsymbol{G}+\boldsymbol{M}(d\boldsymbol{Y}-d\boldsymbol{T}^{real})+(\boldsymbol{m}-\boldsymbol{M}\boldsymbol{e}_{0})B_{ss}dr_{0}^{b}$
- Inflation redistributes from savers to taxpayers, who have higher iMPCs: demand rises
- The deficit no longer has zero present value: $\boldsymbol{q}^{\prime}(d\boldsymbol{G}-d\boldsymbol{T})=-B_{ss}dr_{0}^{b}>0$
- $d\pi_{0}$ comes from the W-NKPC, so again two vector equations

### All together
- Goods: $d\boldsymbol{Y}=d\boldsymbol{G}+\bar{\boldsymbol{M}}(d\boldsymbol{Y}-d\boldsymbol{T})+\bar{\boldsymbol{M}}^{r}d\boldsymbol{r}+\boldsymbol{m}B_{ss}dr_{0}^{b}$
- W-NKPC, rates and taxes as above, with $d\boldsymbol{T}=d\boldsymbol{G}-(\boldsymbol{I}-(1+r_{ss})\boldsymbol{L})d\boldsymbol{B}+B_{ss}(\boldsymbol{L}d\boldsymbol{r}+\boldsymbol{e}_{0}dr_{0}^{b})$
- $\bar{\boldsymbol{M}}^{r}\equiv\boldsymbol{M}^{r}-\frac{p}{1+r_{ss}}\boldsymbol{m}\boldsymbol{q}^{\prime}$
- Solving this gives the same $d\boldsymbol{Y}$ and $d\boldsymbol{\pi}$ as the full model
- An active Taylor rule lowers the multiplier, and nominal bonds raise it
- A fixed rate at the ZLB also raises it (Christiano et al., 2011)

## Unions

### Why sticky wages instead of sticky prices?
- Same $Y$ and $\pi$ for monetary and fiscal shocks: same Phillips curve slope in output
- Firms become trivial
- In the data profits are pro-cyclical. With sticky prices, wages rise more than prices in a boom, so profits fall. With sticky wages, they do not
- Figure: `monpol_irf_sticky_w_p.pdf`

### Flexible wages and MPCs
- Textbook NK model features sticky prices, flexible wages
  - Fine for some questions, but can cause issues in HA models (see Auclert, Bardóczy, and Rognlie (2023))
- EX: Standard HA model where households have flexible labor supply with FOC (EIS=1,Frisch=1): $u'\left(c_{i}\right)= h'\left(l_{i}\right) ,\quad \Leftrightarrow c_{i}^{-1}=\varphi l_{i}$
- What happens if we give household $i$ a lumpsum transfer $\chi$? $-c_{i}^{-2}\frac{\partial c_{i}}{\partial\chi}=\varphi \frac{\partial l_{i}}{\partial\chi}$
- Define $MPC_{i}=\frac{\partial c_{i}}{\partial\chi},MPE_{i}=-\frac{\partial l_{i}}{\partial\chi}$ as the change in consumption and change in earnings following transfer: $MPC_{i}=\frac{c_{i}}{l_{i}}MPE_{i}$
- If HHs have little saving, $c_{i}\approx l_{i}$ $\Rightarrow$ $MPC_{i}=MPE_{i}$!

### Why unions: MPCs and MPEs
- HA model with flexible labor supply implies a tight link between $MPC_{i}$ and $MPE_{i}$
- MPCs in the data: annual estimates between 0.3 and 0.7
- MPEs in the data: range from 0 to 0.04 annually
- So high MPCs imply far too large wealth effects on labor supply
- Solution: unions take households off their labor supply curves
  - Households do not choose $l_{it}$
  - Wealth effects on aggregate $L$ are small, and sticky wages dampen them further

## A canonical HANK model

### A canonical HANK model
- Auclert, Rognlie and Straub (2025): a quantitative version of today's model
- It is the model of the NK framework section, with HANK households:
  - We drop the three assumptions of the IKC: markups, a Taylor rule, nominal bonds
  - We add heterogeneous discount factors
- Your task: use it to compute IRFs and multipliers

### Households
- Log utility, idiosyncratic productivity $e_{it}$ and discount factor $\beta_{it}$:
- Discount factors: $\beta\in\{\beta_{L},\beta_{H}\}$. With probability $q$ per year, a new independent draw
- $r_{t}^{p}$ is the return on the mutual fund

### Calibration
- The model is annual, so with $Y=1$: $A=5$, $B=1$, $p=4$
- $\beta_{H}$ clears the asset market, $\beta_{L}$ raises the MPC
- Table: Real rate, annual \| $2%$ \| Discount factors, $\beta_{L},\beta_{H}$ \| $0.82,1.00$
- Table: Assets to annual GDP \| $500%$ \| Share of patient agents \| $49%$
- Table: Bonds to annual GDP \| $100%$ \| Redraw probability, $q$ \| $0.04$
- Table: Spending to GDP \| $20%$ \| Slope, $\kappa$ \| $0.16$
- Table: Markup, $\mu$ \| $1.11$ \| Frisch elasticity \| $1$
- Table: Taylor coefficient, $\phi$ \| $1.5$ \| Annual MPC, $M_{0,0}$ \| $0.47$
- Table: Debt persistence, $\rho_{B}$ \| $0.90$ \| \|

### Exercise: solve the canonical HANK model
- The code is in Exercise-Canonical-HANK. Shock: $dG$ with $dB_{t}=\rho_{B}(dB_{t-1}+dG_{t})$.
- Read blocks.py and check that all the aggregate equations are included.
- Assume a real rate rule and real bonds, with positive markups. Where does the capital gain come from?
- Solve the IRFs and the impact and cumulative multipliers under the real rate rule.
- Do the same with a Taylor rule and nominal bonds.
- Do it again with less rigid wages: raise $\kappa$.
- Do it again with a stronger Taylor rule: raise $\phi$.

## Summary

### Fiscal policy in HANK: literature
- Seminal paper: Intertemporal Keynesian Cross
- Many other interesting papers:
- McKay and Reis - The role of automatic stabilizers in the US business cycle (2016)
  - Analyze the role of automatic stabilizers in a HANK model
- Bayer, Born, Luetticke - The liquidity channel of fiscal policy (2023)
  - Role of liquid and illiquid effects of fiscal policy
- Hagedorn, Manovskii, and Mitman - The fiscal multiplier (2019)
  - Systematic evaluation of multiplier in HANK + ZLB
- Druedahl, Ravn, Sunder-Plassmann, Sundram, \& Waldstrøm - Fiscal Multipliers in Small Open Economies With Heterogeneous Households (2024)
  - Generalize to small open economies

### Summary and next week
- Today:
  - Same macro blocks, three household blocks, one object $\boldsymbol{M}$
  - RANK: deficits are neutral. TANK: amplification on impact only. HANK: larger and more persistent
- Next week: Assignment workshop
- Homework:
  - Work on exercise
  - Work on assignment

---

# Work needed outside the slides

- Common calibration for RANK and HANK: $\nu$, $\kappa$, $\mu$, $G_{ss}$, $B_{ss}$, $r_{ss}$
- RANK code: government with $B_{ss}>0$ and $T_t=\tau_tY_t$, remove TFP, $\beta$ shock and ZLB, add the real rate rule $1+i_t=(1+r_{ss})(1+\pi_{t+1})$
- TANK code: RANK with a hand-to-mouth share, $C_t=(1-\lambda)C_t^{R}+\lambda Z_t$ with Euler on $C_t^{R}$
- Rewrite the RANK exercise and its solution notebook
- `Exercise-HANK` has no $(1-\tau_t)$ in its W-NKPC, unlike the slides and `Exercise-Canonical-HANK`
- Rerun the figures after the common calibration: `slide_figures.py` in both solution folders
