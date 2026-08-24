# Outline of the slides

Generated from the `.tex` decks at the top level of the repo (the live 2026 edition).
One `#` per lecture, one `##` per `\section`, one `-` per frame.
The exercise sheets (`07-Wealth-Inequality/exercise/HANC.tex`, `08-Secular-Stagnation/exercise/HANC.tex`, `Assignments/`) are not slides and are not listed.

Note: `07-Wealth-Inequality/tex/` only holds `Wealth-Inequality_handout.tex`; the main deck is missing, so lecture 07 below is read from the handout.


# 01. Introduction (JD)

Source: `01-Introduction/Introduction.tex`

- Title slide
- Plan (table of contents)

## Introduction

- Introduction: teacher, the four central economic questions, Python as the technical method, prerequisites
- Macroeconomic Models with Heterogeneous Agents: model components, complete vs incomplete markets, ex ante vs ex post heterogeneity, HANC vs HANK
- History of heterogeneous agent macro: four survey references (Heathcote et al. 2009, Kaplan-Violante 2018, Cherrier et al. 2023, Auclert et al. 2025)

## Structure

- Teaching method: lecture format, content, web and git links, GEModelTools
- Assignments and exam: three assignments with deadlines, feedback, 36-hour take-home
- Python: assumed knowledge, Anaconda, packages, installing GEModelTools
- Course plan: lecture blocks 1-2, 3-6, 7, 8-13, 14

## Learning goals

- Knowledge: six official learning outcomes
- Skills: five official learning outcomes
- Competencies: two official learning outcomes

## Programming in Python

- Classes: attributes, methods, inheritance, code example
- References: `=` assigns a reference, not a copy; slicing and indexing quiz
- Types and in-place operations: atomic types, containers, mutables vs immutables, in-place quiz
- Functions and scope: functions as objects, side effects, recursion, decorators, local vs global scope quiz
- Conditionals and loops: comparison, `if`/`for`/`while`, convergence, `for...else` quiz
- Decimal numbers are not exact: floating point, order of computation, underflow/overflow quiz
- Pseudo random numbers: seeds, generator state, Monte Carlo vs quadrature vs transition matrix
- Documentation and debugging: writing docs, design patterns, run top to bottom, debugging advice
- Numba and EconModelClass: what each package gives you

## Consumption-Saving

- Consumption-saving: pointer to the lecture 2 slides


# 02. Consumption-Saving

Source: `02-Consumption-Saving/Consumption-Saving.tex`

- Title slide

## Introduction

- Introduction: three generations of consumption models, life-cycle evidence, empirical MPCs, Jappelli-Pistaferri book
- Plan (table of contents)

## PIH

- Consumption-saving: the deterministic finite-horizon problem, variables and parameters
- It is a *static* problem: information, target, behavior, solution as a sequence of choices
- IBC: substitution into the intertemporal budget constraint, human capital and after-interest assets
- FOC and Euler-equation: Lagrangian, first order conditions, Euler equation
- Consumption choice: CRRA, when consumption is constant, closed-form $c_0^*$
- Infinite horizon $T\to\infty$: limit formula, constant income case, annuity value when $\beta R=1$
- Propensities to consume: MPC out of windfall, future and permanent income; dynamic effects
- Savings ($\beta R=1$): constant savings, effect of $\beta R \lessgtr 1$, savings change with income timing
- Empirical MPCs: figure from Fagereng et al. (2021)
- Initial liquidity/borrowing constraint: hard constraint, maximum consumption, MPC of one when constrained

## Buffer-stock

- Uncertainty and always borrowing constraint: stochastic income, no-Ponzi, a true dynamic problem
- IBC: infinite-horizon budget constraint and the transversality argument
- Natural borrowing limit: derivation of $a_t \geq -w\underline{z}/r$
- Euler-equation from variation argument: two cases, constrained vs unconstrained, sufficiency
- Special case I: Quadratic utility: certainty equivalence, consumption is a random walk
- Special case II: CARA utility: closed-form consumption function, precautionary saving
- Dynamic solution, Bellman's Principle of Optimality: value function, policy function, FOC and envelope
- Vocabulary: state, control, continuation value, parameters
- Infinite horizon $T\to\infty$: contraction mapping, backward iteration in practice

## 3-periods

- 3-period model: utility, income, cash-on-hand, borrowing constraint, three-state productivity process
- Bellman equation: the recursion with $v_3=0$
- Discretization: grids for $z$ and $a$, expectation as a sum, ConSav tools
- Grids and transition probabilities: figures, size of risk scaled by $\Delta$
- Linear interpolation: the formula, `linear_interp.interp1d`
- Linear interpolation: figure
- Value function iteration (VFI): value-of-choice, inner and outer loop, numerical optimizers
- Consumption function: figure
- Savings function: figure
- Change in savings function: figure
- MPC: figure
- Intertemporal MPC: figures, no wealth effect since $r=0$
- Economic insights: concave consumption, precautionary motive, MPC decreasing in assets, buffer-stock target
- Numerical Monte Carlo simulation: initial distribution, simulation steps, pros and cons
- Taking stock: what VFI gives you and its bottleneck

## EGM

- Time iteration: replace optimization with root-finding on the Euler equation
- Endogenous grid-point method (EGM): post-decision marginal value, invert Euler, endogenous cash-on-hand, constrained vs interior

## Income process

- Permanent transitory income process: persistent plus transitory shocks, normalization, Gauss-Hermite
- Transition probabilities: quadrature for $\xi$, Tauchen/Rouwenhorst for $\tilde z$, tensor and Kronecker products

## Full household problem

- Full household problem: infinite-horizon problem with AR(1) log productivity
- Recursive formulation I: value function, same Euler, constrained consumption
- Recursive formulation II: beginning-of-period value function, one interpolation per guess
- Distribution: why Monte Carlo hurts market clearing, the histogram alternative, $\Pi_z$ and $\Lambda$
- Numerical histogram simulation: distribute stochastic mass, then endogenous mass with interpolation weights
- Implementation: toy example spreadsheet, comparison with Monte Carlo

## Misc

- 1. Life-cycle (I): basic structure, add-ons, Jorgensen (2017) as a starting example
- 1. Life-cycle (II): Gourinchas and Parker, buffer-stock vs retirement saving
- 1. Life-cycle (III): inheritance natural experiment, Druedahl and Martinello (2022)
- 2. More realistic income risk (I): non-lognormal earnings changes, Guvenen et al. (2021)
- 2. More realistic income risk (II): monthly zero-growth mass, Druedahl et al. (2021)
- 3. Epstein-Zin: recursive preferences, separate IES and risk aversion, Euler and envelope
- 4. Deep learning: curse of dimensionality, neural network solvers, EconDLSolvers

## Summary

- Summary and what's next: recap of the four blocks, next is stationary equilibrium, read Aiyagari (1994)


# 03-04. Stationary Equilibrium (RH)

Source: `03-04-Stationary-Equilibrium/tex/Stationary-Equilibrium.tex`

- Title slide
- Recap of the two last classes: buffer-stock model, VFI vs EGM, Monte Carlo vs histogram

## Introduction

- Introduction: from partial to general equilibrium, HANC, stationary equilibrium, GEModelTools, Aiyagari (1994)
- Outline of this lecture: Ramsey recap, HANC, computing the stationary equilibrium, economic properties

## Ramsey-recap

- The Ramsey model: HANC merges Ramsey-Cass-Koopmans with the one-asset buffer-stock model
- Ramsey: Firms: production function, profits, rental rate and wage, zero profits under CRS
- Ramsey: Zero-profit mutual fund: role, depreciation, return on deposits, balance sheet
- Ramsey: Households: utility maximization, budget constraint, Euler equation
- Ramsey: Market Clearing: capital, labor, goods markets and Walras' law derivation
- Ramsey: Summary: simplified and extended form of the model
- Ramsey: As an equation system: eight equations per period with the list of unknowns
- Ramsey: Steady state: solve Euler for $r_{ss}$ and $K_{ss}$; asset supply is inelastic

## HANC

- HANC model overview: firms, mutual funds, households, markets; other names for the model
- Heterogeneous households: the household problem, where heterogeneity enters, incomplete markets
- Recursive formulation: value function at decision
- Distributions and aggregates: policy functions, the distribution, integrating out individual states
- Equation system: full system, size comparison with Ramsey (2108 vs 8 equations per period)
- Market clearing: capital, labor, goods; Walras' law via Euler's theorem

## Computing the Stationary Equilibrium

- Stationary equilibrium - equation system: the steady-state system, law of large numbers note
- Stationary equilibrium - more verbal definition: quantities, prices, distribution, policies and the seven conditions
- How do we solve the household block in practice?: EGM backward, histogram forward, aggregate
- Direct implementation (K guess): root-finding in $K_{ss}$, five steps
- Direct implementation (r guess): root-finding in $r_{ss}$, five steps
- Indirect implementation: choose $r_{ss},w_{ss}$ and back out $\Gamma_{ss}$ and $\delta$
- Direct implementation (calibration): target $r$, $K$, $Y$ and root-find in $\beta$
- How to choose parameters?: external vs internal calibration, informal, formal, estimation
- Calibration at the quarterly level: the five target values used

## Some Properties of the HANC steady-state

- Consumption function: Euler still necessary, precautionary saving, buffer-stock target, high MPC
- Some amount of inequality: marginal distributions over $z$ and $a$; income shocks drive wealth inequality
- Steady state interest rate: no aggregate Euler with heterogeneous agents, market clearing figure
- Risk drives wealth accumulation: $\sigma_\psi$ vs aggregate assets figure
- Marginal Propensity to Consume: MPC policy figure
- Tradeoff between matching aggregate wealth and MPCs: figure, and $\beta$-heterogeneity as the fix

## Exercises

- Exercise 1: HANC with ex-ante heterogeneity: three $\beta$ types, find $\delta^\beta$ hitting MPC = 0.27 and K/Y = 16
- Exercise 2: HANCGovModel: endowment economy with government bonds, taxes and a tax rule
- Exercise 2: Households: household problem, Euler equation, envelope condition
- Exercise 2: Questions: five questions ending with average utility maximization


# 05-06. Transition Path (RH)

Source: `05-06-Transition-Path/tex/Transition-Path.tex`

- Title slide

## Introduction

- Introduction: from stationary equilibrium to transition paths; Auclert et al. (2021), Kirkby (2017)
- Outline: six items from the Ramsey transition to first-order approximations
- Example I: permanent labor supply shock in the Ministry of Finance MAKRO model
- Example II: temporary public spending shock, back to the same steady state

## Ramsey model

- Ramsey: Summary: simplified form, functional forms, steady state
- Ramsey: As an equation system: eight equations, perfect foresight, list of unknowns
- Recap: Newton's method I: first-order Taylor approximation, the update rule
- Recap: Newton's method II: iteration, numerical derivative, convergence properties
- Recap: Multivariate Newton's method: vector-valued update with the Jacobian
- Recap: Broyden's method I: why to avoid recomputing $J$, the rank-one update
- Recap: Broyden's method II: the seven-step algorithm
- Back to Ramsey: the two problems, many unknowns and infinite time
- Truncated Ramsey, reduced vector form: reduce to two residual equations, truncation argument
- Further reduced: reduce to one residual in $\boldsymbol{K}$
- Sequence space: what it means, Keynesian consumption function example, non-linear and forward-looking cases
- Solution in sequence space: $T=200$, numerical Jacobian, Broyden; transitory vs permanent transitions
- Example 1: permanent from low capital: $K_{-1}=0.75K_{ss}$ figure
- Example 2: transitory following technology shock: AR(1) TFP shock, MIT-shock terminology

## Transition path in PE

- Household model in a transition: the HANC household block with time-varying $r_t,w_t$
- Perfect foresight, initial and terminal conditions: the four assumptions
- Impulse responses: backward and forward step: goal, then the two steps
- Summing up the transition in PE: three inputs, two steps, then aggregate to get the IRF
- (untitled) "Let's code!"

## Transition path in GE

- Equation system: full GE system including the distribution transitions
- Transition path - close to verbal definition: quantities, prices, distributions, policies and the seven conditions
- Reduce size of equation system: why a $(T\times N)$ Jacobian is expensive
- Truncated, reduced vector form: residuals in $A^{hh}-A$ and $L^{hh}-L$
- DAG - Directed Acyclic Graph: shocks, unknowns, blocks and the implied evaluation order
- Further reduction: single residual $A^{hh}(w(K),r(K))-K$
- Solve with Broyden: $T$ equations and $T$ unknowns
- How to compute Jacobian?: use the DAG and the chain rule instead of differentiating the whole model
- What is a Jacobian: $H_K$ by the chain rule, the structure of $\mathcal{J}^{A^{hh},r}$, economic reading of rows and columns
- How to compute Jacobian?: cheap firm-side Jacobians vs the costly household Jacobians
- Bottleneck: How do we find the Jacobian?: the naive $T^2$ approach, pointer to the fake news algorithm
- Summary: five steps to compute a transition path
- (untitled) "Let's code!"
- Reminder: PE impulse responses, GE market clearing, Newton and the sequence-space Jacobian

## Fake News Algorithm

- Fake news algorithm: household block notation, the goal, why the naive approach is $T^2$
- Initial step: aggregate as policy times distribution, linearize into a policy term and a distribution term
- Pertubation of policy function: the shift property $y^s_t = y^{s+j}_{t+j}$, only relative time matters
- Numerical illustration: response of $c_t$ to income shocks at $s=3,5$
- Implementation: five-step algorithm to build $d\boldsymbol{A}^{hh}/d\boldsymbol{w}$
- (untitled) "Let's code!"
- Fake news algorithm - summary: Auclert et al. (2021), central insight, how it chains along the DAG, automated in GEModelTools

## Linear transitions and aggregate uncertainty

- Reminder of model class: unknowns, shocks, target system, the deterministic MIT-shock solution
- Aggregate uncertainty: stochastic shocks make all variables random, expectations in forward-looking equations
- Stochastic vs deterministic models: first-order Taylor shows aggregate uncertainty does not matter to first order
- Linearized IRFs: $d\boldsymbol{U}=-H_U^{-1}H_Z d\boldsymbol{Z}$ and the three limitations
- Linearized IRFs: example from HANC: $H_K$, $H_\Gamma$ and the matrix-product solution
- Simulating a time-series using the linearized solution: shocks as $MA(\infty)$, three-step simulation
- (untitled) "Let's code!"
- When does aggregate uncertainty matter?: Boppart et al. (2018) equivalence, second-order approximation and $\sigma^2_{x,t}$
- Calculating moments - variance: three steps to get $\text{var}(dC_t)$
- Calculating moments - covariance: covariance formula and the shock decomposition
- Solving HA model with aggregate risk (advanced): state space vs sequence space
- Example: Krusell-Smith: HANC with aggregate TFP risk, $\boldsymbol{D}_t$ as a state variable
- Comparisons: state-space linearization, global solution, deep learning, discrete aggregate risk
- Summary: four things to remember

## Exercises

- Exercises: HANCGovModel: six questions, transition path, DAG, Jacobians, $G$ shock, savings, consumption inequality


# 07. Wealth Inequality (RH)

Source: `07-Wealth-Inequality/tex/Wealth-Inequality_handout.tex` (handout; main deck missing)

- Title slide
- Summary of the class so far and next steps: end of the computational part
- Summary of the class - Steady state: backward and forward steps, root finding in GE, the $\beta$ calibration trick
- Summary of the class - Transitions: PE backward/forward, sequence-space Jacobian, fake news
- Summary of the class - Transitions: transitory vs permanent shocks, non-linear vs linear methods
- Roadmap for the rest of the class: wealth inequality, then secular stagnation, then HANK

## Introduction

- Wealth inequality: goal, three central questions, plan for the day

## Wealth inequality in the data

- Earnings and wealth inequality: 1989 SCF top shares table, wealth more concentrated than earnings
- Wealth more concentrated than earnings: cross-country Gini figure
- Top wealth shares in the US over time: Saez and Zucman (2020) Figure 2
- Income inequality has increased since the 70s (US): Saez and Zucman (2020) Figure 3
- Income growth by decile in the U.S.: Saez and Zucman (2020) Figure 4
- Average tax rates by income groups: Saez and Zucman (2020) Figure 5
- Richer households hold more risky assets: Bach et al. (2020) Figure 2
- Richer households have higher returns: Bach et al. (2020) Figure 3, risk vs skill debate
- The rich save more, because of capital gains: Fagereng et al. (2019) Figure 1
- Taking stock: five stylized facts

## Explaining wealth inequality

- Aiyagari Model: preferences, budget and borrowing constraint, calibrated earnings process
- Aiyagari Model - wealth inequality fit: table comparing data and baseline model Gini and top shares
- Top wealth inequality: Pareto tails, log-linear tail plot
- Policies in the Buffer-Stock model: policy function figure
- Key mechanism: precautionary saving, dissaving above the buffer-stock target, conflict with the evidence
- Explanations: income inequality, preference heterogeneity, bequests, entrepreneurship, return heterogeneity
- Bequests: warm-glow bequest term added to the problem
- Explaining wealth inequality: heterogeneous preferences ($\beta_i$, $\sigma_i$)
- Explaining wealth inequality: entrepreneurship
- Explaining wealth inequality: idiosyncratic rates of return

## Gaillard, Hellwig, Wanger and Werguin (2024)

- Ranking of Pareto tails in the Data: capital income < wealth < labor income < consumption
- Theoretical results: non-homothetic preference for wealth plus scale-dependent returns
- Quantitative model overview: household ingredients plus a standard Cobb-Douglas supply side
- Quantitative results - Bellman equation: the full recursion with taste for wealth, HSV taxes, scale dependence
- A note on non-homothetic taste for wealth: wealth as a luxury good when $\nu<\gamma$
- A note on death probability: perpetual youth, accidental bequests, why it is needed for a non-degenerate distribution
- Quantitative results - details on heterogeneous returns: worker vs entrepreneur type, Markov switching rates
- Quantitative results - supply side and market clearing: production, asset market clearing, government budget
- Quantitative results - main exercise: five estimated parameters, three targeted Pareto ratios plus extra targets
- Results: Table 4; non-homotheticity and type dependence are both needed for the tail ranking

## Hubmer, Krussel and Smith (2021)

- Explaining wealth inequality: the paper's question, matching 1967 and projecting forward
- Model: household problem with non-linear taxes, heterogeneous returns and $\beta$-heterogeneity
- Facts: Fagereng et al. (2020) evidence on return heterogeneity, persistence and correlation with wealth
- Equilibrium: capital market clearing: two equilibrium objects, $K_t$ and $\underline r_t$
- Calibration strategy summary: calibrate to observables, set $\beta$ dispersion residually, feed in changes 1967-2015
- Return heterogeneity: overall return, endogenous risk-free rate, exogenous excess return schedules
- Calibration: return process: asset class decomposition of $r^X$ and $\sigma^X$
- Excess return schedule details: 1967 aggregate excess returns and portfolio weights by percentile
- Schedule of excess returns: figure

### Results

- Results, I: Steady state (1967): top share fit with and without $\beta$-heterogeneity
- Results, I: steady state (1967): decomposition table, return heterogeneity as the key ingredient
- Next step: transition: the four factors fed in
- Observed change 1: Decrease in tax progressivity: Piketty and Saez effective federal tax rates
- Observed change 2: Increase in labor income risk: Heathcote, Storesletten and Violante variances
- Observed change 3: Increase in top labor income shares: Pareto tail coefficient from 2.8 to 1.9
- Observed change 4: return premia: figure (1 of 3)
- Observed change 4: return premia: figure (2 of 3)
- Observed change 4: return premia: figure (3 of 3)
- Results, II: historical evolution: model vs data wealth shares over time
- Results: Capital-output ratio and bottom 50%: figure
- Results: Risk-free rate: endogenous $r$ matches level and decline
- Decomposition of transitional dynamics: figure
- Decomposition of transitional dynamics: declining tax progressivity explains the rise; return premia give the U-shape; subtle role of earnings dispersion
- Summary: model ingredients and the two main findings
- Ozkan et al. (2024): lifecycle decomposition with Norwegian admin data, five contributions
- Results from Ozkan et al. (2024): decomposition for the top 0.1% and for "new money"

## Application to Wealth Taxation

- Wealth taxation I: why tax wealth (redistribution) and why not (two efficiency concerns)
- Wealth taxation II: Guvenen et al. (2023), wealth tax vs capital income tax, equivalence without return heterogeneity
- Taxation with return heterogeneity: capital income tax is distortionary, wealth tax shifts the base to unproductive agents
- Model: household problem, the two tax systems, entrepreneurial ability and profit
- Empirical fit: model reproduces the Pareto tail of US wealth
- Results: revenue-neutral swap to a 1.2% wealth tax raises K, output, wages; ~7% welfare gain
- Results - optimal taxation: optimal wealth tax 3.03% (9% gain) vs optimal capital income subsidy -13.6% (4.2% gain)
- Summary: source of inequality matters for optimal taxation

## Exercise

- Standard HANC model with return heterogeneity: household problem, three questions (solve PE, calibrate to 4% returns, compare distributions)

## Summary

- Summary and next week: explanations of wealth inequality, secular stagnation next, midterm evaluation, homework


# 08. Secular Stagnation (RH)

Source: `08-Secular-Stagnation/tex/Secular-Stagnation.tex`

- Title slide

## Introduction

- Secular Stagnation: today's question, three central questions, two candidate explanations
- Secular Stagnation: definition, and the claim that advanced economies have been there ~20 years
- Declining interest rates: Mian, Sufi and Straub (2021) rates figure
- Declining growth: trend growth figure
- Rising saving rates and declining $r$: Mankiw (2022) Solow arithmetic, $dr/ds$ vs $dr/d(n+g)$, ZLB implications
- Explanations: market power, relative price of investment, aging, income inequality

## Demographics

- The world population is aging: figure
- ...wealth-to-GDP ratios are increasing...: figure
- ...rates of return on wealth are falling...: figure
- ...and "global imbalances" are rising: figure
- How have demographics shaped these trends?: agreement on direction, disagreement on magnitude
- And how will demographics continue to shape these trends?: Lane (ECB), asset market meltdown, great demographic reversal
- And how will demographics continue to shape these trends?: Auclert, Malmberg, Martenet and Rognlie (2021), multi-country OLG
- Taking the model to the data: the sufficient statistic approach and its four inputs
- Main results: rejects the great demographic reversal and the asset market meltdown hypotheses

## Model

- Model: Main Elements: OLG structure, demographics, CES production, government budget
- Environment: heterogeneous agents: cohort problem, EIS, age-specific discounting, survival, taxes and transfers
- Equilibrium: individual and firm optimization, global asset market clearing, aggregate wealth by country
- Compositional effects as sufficient statistics: Proposition 1 for $W/Y$ in a small country aging alone
- Compositional effects as sufficient statistics: $\Delta_t^{comp}$ definition, why it is measurable, link to GE

## Measuring compositional effects

- Measuring $\Delta^{comp}$: computed for 25 countries, data sources, implied level change
- $\Delta^{comp}$ in the United States: 1950-2100: composition effect raises W/Y (figure 1 of 2)
- $\Delta^{comp}$ in the United States: 1950-2100: figure 2 of 2
- Where do these large effects come from?: is it wealth or income? (question only)
- Where do these large effects come from?: figure 7b
- Where do these large effects come from?: figure 7c
- Where do these large effects come from?: figure 7c with the W vs Y split (2/3 vs 1/3)
- Where do these large effects come from?: breakdown figure, the demographic dividend and its reversal
- Across countries, $\Delta^{comp}$ large and heterogeneous by 2100: figure using US profiles
- General equilibrium: from fixed $r$ and fixed behavior to endogenous $r$
- General equilibrium implications: asset demand and supply semielasticities defined (figure 9a)
- General equilibrium implications: asset demand shift $\bar\Delta^{comp}$ (figure 9b)
- General equilibrium implications: figure 9c
- General equilibrium implications: Proposition 2, $\Delta r \approx -\bar\Delta^{comp}/(\bar\epsilon_S+\bar\epsilon_d)$
- What determines the asset demand semielasticity?: substitution and income components, calibrated values
- What determines the asset supply semielasticity?: $\bar\epsilon^s$ formula, user cost, capital-wealth ratio
- Change in world interest rate: sign argument and the $\sigma \times \eta$ table
- Change in capital to income ratio: formula and the $\sigma \times \eta$ table
- Change in net foreign assets: figure 9d, heterogeneity across countries
- Change in net foreign assets: fast-aging countries lend abroad, the NFA formula
- Change in net foreign assets: figure 10, large global imbalances going forward
- Limitation to baseline model: five limitations of the sufficient-statistic analysis
- Results from richer model: what the compositional effect is a product of
- Results from richer model: Table 1
- Results from richer model: Table 2, GE effects similar
- Conclusion: compositional effect is what matters; refutes meltdown, W/Y keeps rising, larger imbalances

## Income Inequality: Straub (2019)

- Secular stagnation and income inequality: parallel literature, redistribution to high savers lowers $r$
- Stylized model: one-period lived households with utility over wealth, $c_t \approx k(a_t+w)^\phi$
- Empirical estimate: interpretation of $\phi \lessgtr 1$, the two questions
- Empirical estimate: PSID regression, binned scatter, $\phi \approx 0.7$
- Permanent redistribution in the canonical HA model: standard HA problem with a permanent income state, homothetic preferences
- Homothetic household problem I: normalizing constraints by $p$ and the Bellman by $p^{1-\sigma}$
- Homothetic household problem II: scale independence implies $\phi=1$ and no aggregate effect of redistribution
- Non-homothetic HA model: add a taste for wealth; non-homothetic when $\sigma \neq \sigma_a$
- Applications: calibrate to $\phi=0.7$, three permanent income groups, feed in Piketty-Saez inequality, solve GE
- Results: Straub (2019) Figure 6
- GE implications of rising income inequality: homothetic vs non-homothetic, the fixed-K Lucas-tree case
- Trickle down economics?: perfect competition implies gains for all; with markups, savings may only raise asset prices

## Mian, Sufi, Straub (2021) Indebted Demand

- Introduction: household debt and returns, inequality plus deregulation lower $r$, why that is counterintuitive
- Model: continuous time endowment economy, savers with wealth in utility, collateral-constrained borrowers, equilibrium
- Non-homothetic model: calibration to the top 1%, downward-sloping saving schedule, the $\bar c^s = r_t a_t^s$ intuition
- Results: inequality and deregulation both raise debt and lower rates
- Results: inequality shifts the supply curve left (figure)
- Results - Monetary and fiscal policy: permanent deficit shock, short-run vs long-run $r$, ZLB implications
- Exercise: PE HA model with taste for wealth; derive the Euler, update EGM, solve three parameterizations, redistribution experiment

## Summary

- Summary and next week: recap, business cycles from now on, NK model and HANK next


# 09-10. Fiscal Policy in HANK (RH)

Source: `09-10 HANK-Fiscal/tex/HANK-Fiscal.tex`

- Title slide

## Introduction

- Introduction: from long-run trends to business cycles; NK and solution-method literature
- Business cycles: FRED volatility figure, plan for the rest of the course, the need for monetary non-neutrality
- New Keynesian framework: demand-determined output, monopolistic competition, price rigidities

## Quick Recap: The IS-LM model

- The IS-LM model: aggregate demand, the consumption function, the reduced-form output equation
- Policies in IS-LM: fiscal and monetary multipliers, direct vs indirect effects, what is missing

## The New Keynesian model

- Model: the three agents in the simplest NK version
- Overview: households, intermediary goods firms, final goods firms, central bank
- Households: representative household problem and first-order conditions
- Households: forward-looking behavior, Ricardian equivalence
- Final goods firms: CES aggregator, static problem, demand curve, zero profits
- Intermediary goods firms: Rotemberg price adjustment, dynamic problem, symmetry, NKPC, dividends
- Derivation of NKPC: FOC with respect to $p_{jt}$, envelope condition, symmetry
- Central NKPC intuition: zero-inflation steady state, role of $\kappa$, expected future inflation
- Government and central bank: Taylor rule, Fisher relationship, bonds in zero net supply
- Equilibrium: goods, labor and asset market clearing; impose two of three
- Aggregate shocks: TFP, discount factor and monetary policy processes
- The 3 equation NK model: zero-inflation steady state, linearization, IS/NKPC/Taylor
- Technology shocks in the NK model: IRF figure and the transmission chain
- Monetary policy shocks in the NK model: easing IRF figure and its mechanism
- Monetary neutrality: role of sticky prices, IRFs across NKPC slopes
- Some limits of the NK model: no fiscal role, not really Keynesian, no redistribution, hard to calibrate to micro data
- Review questions: demand shock effects, markup cyclicality, government spending and financing

## Exercise

- Exercise - NK model with government: five questions, non-linear vs linear IRFs, adding a ZLB, stabilization policy

## HANK: Introduction

- Introduction: the canonical HANK model with sticky wages; Auclert et al. (2023) intertemporal Keynesian cross

## Sticky Wages

- Detour: early HANK papers just swapped RA for HA, and why that has undesirable properties
- Sticky wages: flexible labor supply ties MPC to MPE; the derivation
- MPCs and MPEs: empirical MPCs (0.3-0.7) vs MPEs (0-0.04), the tension
- Union setup: take households off their labor supply curve via unions; Erceg, Henderson and Levin (2000)
- Union problem: labor demand for union $j$, union objective with a quadratic wage adjustment cost
- New Keynesian Wage Phillips curve: the NKWPC, its four comparative statics, how it breaks the wealth effect
- More tractable version: NKWPC with $u'(C_t)$ instead of the integral of marginal utilities
- Profits: procyclical profits in the data, sticky wages fix the NK counterfactual; IRF figure

## HANK

- Model elements: households, firms, unions, mutual fund, central bank, government
- Households: household problem, active vs union decisions, aggregate consumption function
- Firms: production, profits, $w_t = \Gamma_t$
- Mutual fund and assets: real government bonds, no-arbitrage $r_t = r_t^a$
- Union: everybody works the same, the NKWPC
- Government: budget constraint, tax revenue, the tax rule or a constant-debt rule
- Central bank: nominal Taylor rule plus Fisher, or a direct real rate rule; passive policy special case
- Market clearing: asset, labor and goods markets

## Fiscal Policy

- Simpler consumption function: three assumptions, tax bill, disposable income $Z_t$, sequence-space consumption function
- Side-note: Two-equation version in $\boldsymbol{Y}$ and $\boldsymbol{r}$: goods market clearing plus the firm/NKWPC block, then constant $r$
- Intertemporal Keynesian Cross: total differentiation, $(\boldsymbol{I}-\boldsymbol{M})d\boldsymbol{Y}=d\boldsymbol{G}-\boldsymbol{M}d\boldsymbol{T}$
- Illustration: the IKC written out in matrix form
- iMPC matrix: reading rows and columns of $\boldsymbol{M}$, the quarterly MPC caveat
- iMPCs in the data: Fagereng et al. lottery evidence pins down the first column only
- Perspective: Static Keynesian Cross: old Keynesian version and the $\text{mpc}/(1-\text{mpc})$ multiplier
- NPV-vector: government and household IBCs imply $\boldsymbol{q}'(\boldsymbol{I}-\boldsymbol{M})=\boldsymbol{0}$
- Form of unique solution: $(\boldsymbol{I}-\boldsymbol{M})$ is not invertible; the left-inverse solution; truncation in practice
- Response of consumption: derivation of $d\boldsymbol{C}=\mathcal{M}\boldsymbol{M}(d\boldsymbol{G}-d\boldsymbol{T})$
- Fiscal multipliers: balanced budget multiplier of one, deficit multiplier potentially above one
- Fiscal multiplier: impact and cumulative multiplier definitions
- Comparison with RA model: $\boldsymbol{M}^{RA}=(1-\beta)\boldsymbol{1}\boldsymbol{q}'$, zero consumption response, multiplier of one
- Details on matrix formulation: step-by-step expansion of $(1-\beta)\boldsymbol{1}\boldsymbol{q}'$
- TANK: Campbell-Mankiw two-agent model, $\boldsymbol{M}^{TA}=(1-\lambda)\boldsymbol{M}^{RA}+\lambda\boldsymbol{I}$, four drawbacks
- Comparison with TANK model: the IKC in TANK and the closed-form solution with amplification
- TANK Proof: full derivation of the TANK solution
- Cumulative multiplier still one: proof
- Jacobian columns: columns of $\boldsymbol{M}$ across TANK, HANK and other models
- iMPCs in models: figure
- Multipliers and debt-financing: figure
- Summary in table: IKC Table 1
- Interest rate effects: nominal vs real bonds, surprise inflation and period-0 capital losses
- Generalized IKC: budget constraint with an initial capital gain, real vs nominal bond cases
- Generalized IKC: $d\boldsymbol{C}^{hh}=\boldsymbol{M}^r d\boldsymbol{r}+\boldsymbol{M}(d\boldsymbol{Y}-d\boldsymbol{T})+\boldsymbol{m}^{cap}\text{cap}_0$; capital gains are small
- Fiscal policy in HANK - litterature: McKay-Reis, Bayer-Born-Luetticke, Hagedorn et al., Druedahl et al.

## Exercise

- Exercise: five questions, household Jacobians, deficit vs tax financing, IKC formula check, active monetary policy, flatter NKWPC

## Summary

- Summary and next week: fiscal policy in HANK with sticky wages, assignment workshop next


# 11. Monetary Policy in HANK (RH)

Source: `11-HANK-Monetary/tex/HANK-Monetary.tex`

- Title slide

## Summing Up What We Did So Far

- Aggregate Consumption Function in HANK: assumptions giving $C^{hh}(\boldsymbol{Y}-\boldsymbol{T})$, and why $\boldsymbol{M}$ is enough
- Plan for Today: IKC for monetary policy, KMV (2018), deviations from rational expectations, exercise, assignment time

## Introduction

- Introduction: from fiscal to monetary policy; KMV (2018), Auclert-Rognlie-Straub (2020), Alves et al. (2020)

## Monetary Policy in HANK

- Monetary Policy: heterogeneity changed fiscal transmission; what about monetary policy?
- Model: the canonical HANK model without government, and firm equity as the liquid asset
- Households: household problem with real labor income $Z_t$, the consumption function
- Firms: production, profits, $w_t = 1/\mu$, positive profits in equilibrium
- Mutual fund I: shares, dividends, symmetry, total value of firm equity
- Mutual fund II: the fund problem, FOC for the equity price, ex-post return and valuation effects
- Union: everybody works the same, the NKWPC
- Central bank: nominal Taylor rule with Fisher, or a real rate rule
- Market clearing: asset, labor and goods markets
- The consumption function: linearization into $\boldsymbol{M}d\boldsymbol{Z}+\boldsymbol{M}_r d\boldsymbol{r}+\boldsymbol{m}\,dcap_0$
- Interest rate Jacobians: $\boldsymbol{M}_r$ and $\boldsymbol{m}^{cap}$ figures
- Monetary policy in sequence-space: $d\boldsymbol{Y}=\mathcal{M}\boldsymbol{M}_r d\boldsymbol{r}$, direct vs indirect effects, two questions
- HANK-RANK equivalence: Werning (2015) exact equivalence under log utility, decomposition figure
- HANK-RANK equivalence: same effectiveness but different transmission; the five assumptions behind it

## KMV 2018

- Monetary policy according to HANK: the seminal paper, link to Kaplan and Violante (2014), liquid and illiquid assets
- Household problem: two-asset problem with deposit adjustment costs, wealthy hand-to-mouth
- MPCs: MPCs by stimulus size and across the wealth distribution
- Direct vs indirect effects: HANK amplification, indirect effects ~80% of transmission (Table 7)

## Expectations

- Micro Jumps, Macro Humps: Auclert, Rognlie and Straub (2020), matching the hump-shaped empirical response
- The problem: standard model gives no hump regardless of shock shape
- The solution: RANK: habits generate persistence but kill iMPCs in HANK
- Deviations from alternative expectations: imperfect expectations about aggregates only, first-order implementation
- Income Jacobian: $\boldsymbol{M}$ under rational expectations; only entries above the diagonal involve the future
- Expectations matrix: definition of $\boldsymbol{E}$ and how to read its columns
- Stylized Example I: a concrete $\boldsymbol{E}$, expected paths at $t=0$ and $t=1$, period-0 response
- Stylized Example II: period-1 response decomposed into past shock, present shock and expectation revision
- General formula: $\hat M_{t,s}=\sum_\tau (E_{\tau,s}-E_{\tau-1,s})M_{t-\tau,s-\tau}$
- Examples: $\boldsymbol{E}^{RE}$ vs $\boldsymbol{E}^{Myopic}$ and the implied Jacobians
- Jacobians: $\boldsymbol{M}$ under rational vs myopic expectations (figures)
- Solving GE with non-RE expectations: substitute $\hat{\boldsymbol{M}}$ into the GE solution
- Non-RE expectations in GEModelTools: five-step workaround, overwriting `model.jac_hh`
- Back to Auclert, Rognlie, Straub (2020) - Sticky expectations: Mankiw-Reis sticky information, the implied $\boldsymbol{E}$
- Sticky expectations: properties, $\theta=0$ vs $\theta=1$, iMPCs preserved unlike habits
- Estimation: full HANK model, Romer and Romer shocks, estimated $\theta=0.935$
- RE vs. Non-RE: why sticky expectations are needed to match the data
- Direct and indirect effects: decomposition in the estimated model, indirect effects dominate
- Importance of Investment: investment drives part of the indirect effect
- Summing-Up: three takeaways on using the SSJ machinery with non-RE expectations

## Exercise

- Exercise: four questions, HANK vs RANK decomposition, myopic and sticky expectations, nominal debt with a relaxed borrowing constraint

## Summary

- Summary and next week: monetary policy in HANK, expectations; HANK plus unemployment risk next (JD)


# 12. I-HANK (RH)

Source: `12-IHANK/tex/IHANK.tex`

- Title slide

## Introduction

- Introduction: from closed to small open economy; Auclert et al. (2024), Druedahl et al. (2024) x2

## IHANK Model

- Small Open Economy HANK Model: Gali-Monacelli (2005) plus sticky wages plus heterogeneous agents
- Model components: households, firms, unions, mutual fund with foreign bonds, central bank, foreign economy
- Households: household problem and consumption function
- Consumption basket: CES over domestic and foreign goods, FOCs, CPI
- Aggregate consumption basket: aggregation works because CES preferences are homothetic
- Non-homothetic preferences: what breaks if rich and poor hold different baskets
- Prices: law of one price, nominal exchange rate convention
- Firms: production, profits, FOC linking $P_H/P$ to the real wage
- Mutual fund and assets: equity plus foreign bonds, free capital flows, equity pricing and UIP
- Union: everybody works the same, the NKWPC
- Central bank: floating exchange rate Taylor rule, or a fixed exchange rate
- Foreign Economy: exogenous foreign rate and price, Armington demand for domestic goods
- Trade and current account: GDP, net exports, NFA, current account and the Walras link
- Market clearing: labor market and two equivalent versions of goods market clearing

## International Keynesian Cross

- Sequence-space - goods market: linearize goods market clearing and the two CES demand curves
- Sequence-space - trade elasticity: derive $\chi=\eta(1-\alpha)+\eta^*$ and expenditure switching
- Sequence-space - HHs: household consumption response with the $Q$ term
- Sequence-space - Keynesian Cross: the four channels (interest rate, multiplier, expenditure switching, real income)

## Monetary Policy

- Sequence-space - Keynesian Cross: UIP links monetary policy to the exchange rate
- HANK-RANK equivalence: exact neutrality when $\chi=2-\alpha$; which effect dominates otherwise
- Monetary policy - $\chi=2-\alpha$: output response figure under neutrality
- Monetary policy - $\chi<2-\alpha$: low short-run trade elasticity makes monetary policy less effective in HANK

## Fiscal Policy

- Fiscal Policy: monetary policy is weaker in the open economy; what about fiscal policy?
- Keynesian cross with G: six channels; with constant $r$ it is isomorphic to the closed economy with $\tilde{\boldsymbol{M}}=(1-\alpha)\boldsymbol{M}$
- Fiscal policy in the open economy: arguments for more and for less effective, the $\alpha \to 1$ limit
- Fiscal spending shocks: deficit-financed G shock; similar multipliers, larger C response offset by net exports
- Fiscal spending shocks - openness: multipliers across the import/GDP quartiles of OECD countries

## Foreign Demand Shocks

- Foreign Demand Shocks: from policy to shocks; Druedahl et al. (2024); other open economy shocks
- Motivation: the Keynesian cross with foreign demand, the decomposition of $d\boldsymbol{C}$, comovement in HANK vs RANK
- Empirical estimates of foreign demand shock: 38 OECD countries, trade-weighted foreign economy, sign restrictions
- Spillover effects: local projections of domestic outcomes on the estimated foreign shock
- Why foreign demand shocks?: clean testable implications, no output/inflation tradeoff, contrast with other shocks
- Model: medium-scale two-sector HANK with input-output structure, sticky prices and wages, dynamic trade elasticities
- Household block: household problem with $\beta$ and sector types, the identity Markov matrix for sectors
- Model fit - floating: IRF figure under a floating exchange rate
- Decomposition: $d\boldsymbol{C}$ into interest rate, labor income and capital gain effects
- Model fit - floating /w investment: investment amplifies the HANK response
- Fixed exchange rate: UIP plus Fisher mean the result survives under a peg
- Model fit - fixed: IRF figure under a fixed exchange rate
- Policy: stabilizing aggregate C with monetary vs fiscal policy, sectoral asymmetry table
- Conclusion: monetary policy less effective, fiscal closer to RANK, foreign demand shocks transmit more
- IHANK - litterature: Guo et al., Aggarwal et al., De Ferra et al., Bayer et al.

## Summary

- Summary and next week: SOE HANK today, advanced topics and exam next


# 13. HANK-SAM (JD)

Source: `13-HANK-SAM/HANK-SAM.tex`

- Title slide (with KU/CEBI/DNRF logos)

## Introduction

- Introduction: from RANK to HANK, why MPCs are central, SAM and endogenous fluctuations in idiosyncratic risk, Broer et al. (2024, 2025)

## HANK-SAM

- Overview: intermediate producers, wholesale price setters, final producers, government, central bank, households
- Equilibrium dynamics: incomplete markets, sticky prices, frictional labor market as the three links
- Household problem: the problem with unemployment duration as a state, dividends, transfers, duration-dependent UI
- Income process: income by unemployment duration and the continuous UI share
- Transition probabilities: beginning-of-period value function, the $u$ grid, separation and job-finding transitions
- Aggregation: the two distributions, time-varying transition matrix, searchers, savings, consumption
- EGM: beginning-of-period value function, the EGM step, consumption and savings policies
- Producers: Hiring and firing: job value, vacancy value, free entry
- Labor market dynamics: tightness, Cobb-Douglas matching, job-filling and job-finding rates, law of motion for $u$
- Price setters: intermediate goods price, Phillips curve, flexible price limit, dividends
- Central bank: Taylor rule
- Government: UI expenses, total expenses and taxes, long-term debt budget, tax rule, transfers
- Financial markets: No arbitrage: pricing of government debt and the ex-post real return
- Market clearing: asset and goods markets, plus a Walras' law check
- Shocks, target, unknowns: one shock, seven unknowns, seven targets
- Steady State: zero inflation, calibrating $A$ and $\kappa$, enforcing asset market clearing through $G_{ss}$
- Calibration: eight groups of parameter values (rates, $\beta$ types, matching, producers, price setters, policy, government)
- Steady state analysis: what to look at in steady state and in the household Jacobians
- Policy analysis: 1% government consumption shock, which IRFs to look at, what drives the consumption response

## Stimulus Effects of Common Fiscal Policies

- Motivation and question: countercyclical fiscal policy, policy design varies, which transfer is most cost effective
- A HA-NK-SAM model: the three ingredients, policy setup, two calibration targets
- Overview: six fiscal policy types, five model extensions, link to Broer et al. (2025)
- Model summary: notation, household and firm policy vectors, income process, the three-equation system
- Directed Cycle Graph: model diagram figure
- Directed Cycle Process: the closed-form solution with $\mathcal{G}=(I-M_{SAM}M_{NK}M_{HA})^{-1}$
- Fiscal multipliers: cumulative multiplier definition, the ordering result by direct fiscal cost
- Policy experiment: same output path for different policies, output and tax figures
- Different fiscal multipliers: Table 4, relative multipliers and relative tax responses with the PE/GE split
- Determinants of fiscal multipliers: Table 5

## Endogenous search (*)

- Endogenous search: discrete search choice, search cost, extreme value taste shocks; flagged as advanced and not in GEModelTools
- Discrete search decision: conditional value functions and the logit formula
- Envelope condition: choice probabilities, envelope condition, why monotonicity breaks
- Upper envelope for given $z^{i_z}$: generating candidate points and applying the upper envelope
- Illustration: the upper envelope figure; Druedahl and Jorgensen (2017) $G^2EGM$
- Example: a concrete beginning-of-period value function with a kink
- Next-period values: figure
- Raw values of $c^{i_a}$ and $v^{i_a}$: figure showing the overlaps
- Result after upper envelope: figure
- General problem structure: nested problem with discrete and continuous choices; Druedahl (2021) guide

## Summary

- Summary: what HANK-SAM adds, and the two solution-method takeaways


# 14. Exam and Perspectives (JD)

Source: `14-Exam-Perspectives/ExamPerspectives.tex`

- Title slide
- Plan (table of contents)

## Learning outcomes

- Knowledge, Skills and Competencies: the two things you need to know
- Knowledge: six official learning outcomes
- Skills: five official learning outcomes
- Competencies: two official learning outcomes

## Overview and perspectives

- Single-agent problems (partial equilibrium): solution and simulation methods, four directions forward
- Stationary equilibrium: fixed point problem, calibration, comparative statics, perspectives
- Dynamic equilibrium: Aggregate risk: state-space form, why expectations are hard, DSS vs SSS, moving average form
- Sequence space: state-space linearization, what sequence space assumes, MIT shocks, Boppart et al. (2018)
- GE and Household Jacobians: target system, first-order IRFs, non-linear transitions, how derivatives are computed
- Aggregate consumption function: household problem, consumption function, the $\boldsymbol{M}$ matrices
- Intertemporal Keynesian Cross I: production, taxes, unions, disposable income, market clearing
- Intertemporal Keynesian Cross II: the IKC solution with the left-inverse, uniqueness, iMPCs as sufficient statistics
- From HANC to HANK: what pins down the real rate in each, the canonical HANK structure, why sticky wages
- Transition path: temporary vs permanent shocks, welfare analysis, more complex models, precision issues
- Limitation of sequence space approach: first-order, second-order and full dynamic equilibrium
- Business cycle questions: what heterogeneity changes, investment, four open questions

## Exam

- Exam: format, grading, portfolio, time pressure, what to do if code does not work
- Hand-in: figure
- Preparing for the exam: five recommendations

## GEModelTools

- GEModelTools: file structure and the main API calls
- HANK-sticky-wages\simplified: nine things to do live in the code

## Exam from 2023

- Exam from 2023: walk through the 2023 exam


---

## Notes

- The title numbering inside the decks is out of step with the folder numbering: `05-06-Transition-Path` is titled "4. Transition Path", `07-Wealth-Inequality` is "6. Wealth Inequality", `08-Secular-Stagnation` is "7. Secular Stagnation", `12-IHANK` is "13. I-HANK" and `13-HANK-SAM` is also "13. HANK-SAM".
- Lecture 12 (I-HANK) and lecture 14 are slides only, no code.
