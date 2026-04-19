# gram-diffusion-fp-

# Twelve-Stream Structure of Riemann Zeros: From Gram Diffusion to a Fokker–Planck Equation Driven by the Prime Field

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Empirical discovery of a universal 12-stream structure in the nontrivial zeros of the Riemann zeta function, governed by a stochastic process (Gram diffusion) controlled by the Prime Field, with an asymptotic shape parameter $k_\infty = 12/10 = 1.200$.**

---

## 🔥 Core Discoveries

1. **12-Stream Partition:** Zeros naturally split into 12 streams based on Gram indices $m \bmod 12$. The distribution is non-uniform ($p = 5.4 \times 10^{-7}$) with a systematic excess in class 6 ($+0.81\%$) and deficit in class 7 ($-0.73\%$).

2. **Bound Pair (6,7):** Classes 6 and 7 exhibit strong anticorrelation ($r = -0.65$, $p = 10^{-24}$) and their sum is anomalously stable (variance ratio $0.414$). This reduces the effective number of independent streams from 12 to 10.

3. **Asymptotic Limit $k_\infty = 12/10 = 1.200$:** The gamma shape parameter $k(t)$ evolves from $1.040$ at $t \sim 4 \times 10^4$ to $1.206$ at $t \sim 2.68 \times 10^{11}$. A blind test on high-altitude data gives an error of **0.5%** for $12/10$ vs **7.2%** for $9/8$.

4. **Prime Field Control:** Gram stuckings ($\Delta m = 0$) occur at a mean Prime Field of $+2.58$, while normal jumps ($\Delta m = 2$) occur at $-3.08$ ($p < 10^{-300}$, Bonferroni corrected). The effect is **universal** across all 12 classes.

5. **Hermitian Operator $H$:** Constructed from the empirical transition matrix $M$. Its spectrum correlates with the zeros of $\zeta(s)$ with **$r = 0.9275 \pm 0.0135$** (cross-validated on 2,000 blocks). **100%** of tested blocks have $r > 0.85$.

6. **Hybrid Hilbert–Pólya Operator:** A strict train/test split shows the operator predicts zeros with **mean error 0.49** on the first 12 zeros, outperforming direct extrapolation by a factor of 3. Height-dependent calibration mirrors the evolution of $k(t)$.

7. **Fokker–Planck Equation:** The continuous limit of Gram diffusion is derived and calibrated. The restoring force is $\mu(\phi) \approx -0.985\phi$ (Ornstein–Uhlenbeck process), and the stationary distribution matches the empirical phase distribution with $r = 0.9985$.

---

## 📂 Repository Structure
.
├── README.md
├── LICENSE
├── .gitignore
│
├── core/ # 🔥 Main proofs
│ ├── stats_12streams_final.py # Full statistical analysis of 2M zeros
│ ├── asymptotic_blind_test.py # Blind test of k_inf = 1.200
│ ├── asymptotic_theory_12_10.py # Theoretical derivation of 12/10
│ ├── operator_H_correlation.py # Basic H operator test (first 12 zeros)
│ ├── operator_H_basic.py # Minimal H operator construction
│ ├── operator_H_minimal.py # Ultra-minimal version
│ └── operator_M_raw.py # Raw jump matrix analysis
│
├── validation/ # ✅ Cross-validation
│ ├── operator_H_universality.py # Universality test on 2,000 blocks
│ ├── operator_H_honest_test.py # Strict train/test split (no data leakage)
│ ├── validate_gram_indices.py # Validation of Gram indices file
│ └── validate_bootstrap_k.py # Bootstrap analysis of k estimation
│
├── diffusion/ # 🌊 Gram diffusion & SDE
│ ├── diffusion_restoring_force.py # Restoring force μ(φ) ≈ -0.985φ
│ └── diffusion_SDE_simulation.py # Langevin simulation (KS p = 0.195)
│
├── analysis/ # 📊 Additional analysis
│ ├── stats_12streams_2M.py # Detailed 2M zeros analysis
│ ├── stats_pair_6_7.py # Bound pair (6,7) analysis
│ ├── stats_pair_6_7_micro.py # Microscopic analysis of pair (6,7)
│ └── graph_anticorrelations.py # Anticorrelation network (12 even-odd pairs)
│
├── primes/ # 🔢 Prime Field analysis
│ └── prime_contributions.py # Individual prime contributions
│
├── universal/ # 🌍 L-functions
│ ├── universal_L101_25zeros.py # Pilot check on L-mod-101
│ ├── universal_L101_12streams.py # 12-stream test on L-mod-101
│ └── universal_L3_generate.py # Generate 11,400 zeros of L-mod-3
│
├── asymptotic/ # 📈 Asymptotic limit fitting
│ ├── fitting_AIC_comparison.py # AIC model comparison
│ ├── fitting_odlyzko.py # Parameter fitting on Odlyzko data
│ ├── validate_odlyzko_12streams.py # 12-stream validation on Odlyzko
│ └── validate_odlyzko_minimal.py # Minimal Odlyzko validation
│
├── theory/ # 📚 Theoretical models
│ └── theory_explicit_formula.py # Explicit Riemann–von Mangoldt formula
│
├── experiments/ # 🧪 Control experiments
│ └── experiment_fake_zeros.py # Synthetic data control
│
├── figures/ # 🖼️ All visualizations
│ ├── core/ # Main figures for publication
│ │ ├── anticorrelation_graph_spectrum.png
│ │ ├── final_analysis.png
│ │ ├── fokker_planck_corrected.png
│ │ ├── honest_facts_zeta.png
│ │ ├── strict_theory_test.png
│ │ └── universality_H_operator.png
│ ├── supporting/ # Supporting evidence
│ │ ├── class3_anomaly_analysis.png
│ │ ├── delta_jump_types.png
│ │ ├── final_analysis_with_pair_hypothesis.png
│ │ ├── honest_jump_matrix_analysis.png
│ │ ├── honest_pair_analysis.png
│ │ ├── honest_prime_field_analysis.png
│ │ ├── k_evolution_2M.png
│ │ ├── prime_contributions.png
│ │ └── roc_curve.png
│ └── supplementary/ # Additional technical figures
│ ├── delta_by_lag.png
│ ├── delta_evolution.png
│ ├── delta_spectrum.png
│ ├── drachli_12_lens_analysis.png
│ ├── explicit_formula_check.png
│ ├── fokker_planck_direct_mu.png
│ ├── fokker_planck_verification.png
│ ├── k_vs_sample_size.png
│ ├── lehmer_pairs_analysis.png
│ ├── odlyzko_validation_HONEST.png
│ ├── systematic_M_structure_test.png
│ ├── test2_bootstrap_k_MLE_final.png
│ ├── three_critical_tests.png
│ └── zeta_crystal_model.png
│
├── notes/ # 📝 Research notes and logs
│ └── (various .txt files with test results)
│
└── data/ # 📁 Data instructions (not the data itself)
├── README.md # Where to download zeros
└── generate_gram_indices.py # Script to generate Gram indices

text

---

## 🚀 Quick Start

### 1. Download the Data

- **zeros_2M.txt**: First 2,000,000 zeros from [LMFDB](http://www.lmfdb.org/zeros/zeta/).
- **zero_10k_10^12.txt**: 10,000 zeros at $t \sim 2.68 \times 10^{11}$ from [Odlyzko's tables](http://www.dtc.umn.edu/~odlyzko/zeta_tables/).
- **zeros_L101_25.txt**: 25 zeros of L-function mod 101 (included in repo).

Place the data files in the `data/` directory.

### 2. Generate Gram Indices

```bash
cd data
python generate_gram_indices.py
This creates gram_indices_2M.npy (cached for reproducibility).

3. Run the Core Tests
bash
# Full statistical analysis of 2M zeros
python core/stats_12streams_final.py

# Blind test of k_inf = 1.200
python core/asymptotic_blind_test.py

# H operator correlation (first 12 zeros)
python core/operator_H_correlation.py

# Universality test (2,000 blocks)
python validation/operator_H_universality.py

# Strict honest test (train/test split, no data leakage)
python validation/operator_H_honest_test.py
4. Reproduce the Figures
bash
# Gram diffusion & Fokker–Planck
python diffusion/diffusion_restoring_force.py
python diffusion/diffusion_SDE_simulation.py

# Prime Field analysis
python primes/prime_contributions.py

# Anticorrelation graph
python analysis/graph_anticorrelations.py
📊 Key Results Summary
Test	Result	File
12-stream uniformity	$p = 5.4 \times 10^{-7}$ (χ²), permutation $p < 0.0001$	core/stats_12streams_final.py
Class 6 excess / Class 7 deficit	$+0.81%$ / $-0.73%$	core/stats_12streams_final.py
Bound pair (6,7) anticorrelation	$r = -0.65$, $p = 10^{-24}$	analysis/stats_pair_6_7_micro.py
Variance ratio $N_6+N_7$	$0.414$ (anomalously stable)	analysis/stats_pair_6_7_micro.py
$k(t)$ evolution	$1.040 \to 1.104 \to 1.206$	core/stats_12streams_final.py
$k_\infty$ blind test	$12/10$ error $0.5%$, $9/8$ error $7.2%$	core/asymptotic_blind_test.py
Prime Field control	$p < 10^{-300}$, 9/9 robust	primes/prime_contributions.py
H operator correlation (2,000 blocks)	$r = 0.9275 \pm 0.0135$, 100% > 0.85	validation/operator_H_universality.py
Hybrid operator honest test	Mean error 0.49 (first 12), 0.81 ($t \sim 89,480$)	validation/operator_H_honest_test.py
Fokker–Planck KS test	$p = 0.195$ (cannot reject)	diffusion/diffusion_SDE_simulation.py
Anticorrelation network	12 even-odd pairs, $p = 0.0003$	analysis/graph_anticorrelations.py
Class 3 isolation	Degree 0 in anticorrelation graph	analysis/graph_anticorrelations.py
Lehmer pairs stucking frequency	2.5× higher than average	analysis/stats_pair_6_7_micro.py
🖼️ Key Figures
Figure	Description
https://figures/core/universality_H_operator.png	H Operator Universality: Correlation $r = 0.9275 \pm 0.0135$ across 2,000 blocks.
https://figures/core/final_analysis.png	Complete Dashboard: 12-stream distribution, $k(t)$ evolution, gamma fit.
https://figures/core/honest_facts_zeta.png	9 Empirical Facts: All statistically verified without fitting.
https://figures/core/fokker_planck_corrected.png	Gram Diffusion: Restoring force $\mu(\phi) \approx -0.985\phi$.
https://figures/core/strict_theory_test.png	Blind Test: $k_\infty = 1.200$ vs $1.125$ on $t \sim 10^6$ and $t \sim 10^{11}$.
https://figures/core/anticorrelation_graph_spectrum.png	Anticorrelation Network: 12 vertices, class 3 isolated.
https://figures/supporting/prime_contributions.png	Prime Field Decomposition: Primes $p \equiv \pm 5 \pmod{12}$ dominate.
📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

📚 References
Riemann, B. (1859). Monatsberichte der Berliner Akademie.

Gram, J. P. (1903). Acta Mathematica, 27, 289–304.

Montgomery, H. L. (1973). Proc. Symp. Pure Math., 24, 181–193.

Odlyzko, A. M. (1987). Math. Comp., 48(177), 273–308.

Berry, M. V., & Keating, J. P. (1999). SIAM Review, 41(2), 236–266.

von Mangoldt, H. (1895). Journal für die reine und angewandte Mathematik, 114, 255–305.

📧 Contact
For questions or collaboration: [Your Email]

This work provides overwhelming statistical evidence for a universal 12-stream structure in the Riemann zeros, controlled by the Prime Field and converging to an asymptotic shape parameter $k_\infty = 12/10 = 1.200$. The constructed hybrid operator successfully predicts zeros in a strict cross-validation framework, offering a concrete realization of the Hilbert–Pólya conjecture.




Here is the English translation of the provided text, preserving the structure, formatting, and scientific terminology.

**I. STRUCTURE OF ZEROS**
The zeros of $\zeta(s)$ are divided into 12 streams according to the residue classes of Gram indices $m \bmod 12$. The ratio of average intervals $\langle \Delta_c \rangle / \langle \Delta \rangle = 12.000 \pm 0.055$.

The distribution among the 12 classes is non-uniform ($\chi^2 = 50.35$, $p = 5.4 \times 10^{-7}$; permutation test $p < 0.0001$).

Even-odd asymmetry: even classes are overrepresented ($+0.47\%$), odd classes are underrepresented ($-0.47\%$).

Class 6 — excess $+0.81\%$ (maximum), Class 7 — deficit $-0.73\%$ (minimum).

**II. BOUND PAIR (6,7)**
Classes 6 and 7 are anti-correlated: $r = -0.65$, $p = 2.3 \times 10^{-25}$.

The sum $N_6 + N_7$ is anomalously stable: variance ratio compared to expected for independent streams = $0.414$.

Without the pair (6,7), the remaining 10 streams yield $k \approx 1.09$ — close to the Poisson limit $k = 1.0$.

Effective number of independent streams: $N_{\text{eff}} = 10$. This explains $k_\infty = 12/10 = 1.200$.

**III. EVOLUTION OF THE PARAMETER $k$ AND THE LIMIT $k_\infty$**
The shape parameter $k$ of the gamma distribution grows with height:

$t \sim 4 \times 10^4$: $k \approx 1.040$

$t \sim 1.1 \times 10^6$: $k \approx 1.104$

$t \sim 2.68 \times 10^{11}$: $k \approx 1.206$

The asymptotic limit $k_\infty = 12/10 = 1.200$ is confirmed on Odlyzko's data:

$12/10 = 1.200$: error $0.5\%$

$9/8 = 1.125$: error $7.2\%$

95% confidence interval for $k_\infty$: $[1.198, 1.277]$ — includes $1.200$, excludes $1.125$.

**IV. PRIME FIELD CONTROLS GRAM DIFFUSION**
Stalls ($\Delta m = 0$) occur with positive Prime Field: $+2.58 \pm 2.27$.

Normal jumps ($\Delta m = 2$) occur with negative Prime Field: $-3.08 \pm 2.30$.

The difference is statistically significant at $p < 10^{-300}$ (with Bonferroni correction).

The effect is robust across all 9 tested parameter configurations (max_power $\in \{2,3,4\}$, n_primes $\in \{100,200,500\}$).

The most significant contribution comes from primes $p \equiv \pm 5 \pmod{12}$ ($p = 5, 11, 17, 29, 41, \dots$).

Logistic regression on prime contributions predicts event type with AUC = 0.9967.

**V. PHASE SHIFT $\Delta$ AND JUMP TYPE**
Normal jumps ($+2$): mean $\Delta = -0.2743 \pm 0.1665$.

Anomalous jumps ($+1$): mean $\Delta = -0.2927 \pm 0.1570$.

Difference is significant: $p = 4.61 \times 10^{-242}$ (Mann–Whitney U).

**VI. HERMITIAN OPERATOR $H$**
Transition matrix $M$: $P(+1) = 0.584$, $P(+2) = 0.416$. Non-zero entries only for $c \to c+1$ and $c \to c+2 \bmod 12$.

Hermitian operator $H = (M + M^T)/2$ has a spectrum correlating with the zeros of $\zeta(s)$.

On the first 12 zeros: $r = 0.8892$.

On 2,000 blocks of 12 zeros (fine grid): mean $r = 0.9275 \pm 0.0135$. 100% of blocks have $r > 0.85$.

On 17 blocks (coarse grid): mean $r = 0.9238 \pm 0.0145$. 100% of blocks have $r > 0.85$.

Universality confirmed: the spectrum of $H$ correlates with any block of 12 zeros across the entire height range.

**VII. EXACT SCALE INVARIANCE OF MATRIX $M$**
The trace of $M^k$ grows linearly with $k$: $\text{Tr}(M^{11}) = 5.288902$, $\text{Tr}(M^{89}) = 42.792027$.

Trace ratio: $\text{Tr}(M^{89}) / \text{Tr}(M^{11}) = 8.090909\ldots = 89/11$ (match to 6 decimal places).

For any prime $p$ coprime to 12: $\text{Tr}(M^p) \propto p$.

This proves $\mathbb{Z}_{12}$-symmetry and a nearly degenerate dominant subspace.

**VIII. NETWORK OF ANTI-CORRELATIONS**
12 significant anti-correlations found between classes ($p < 0.05$).

All 12 pairs are even-odd ($p = 0.0003$, Monte Carlo test).

Strongest pair: (10,11) with $r = -0.73$.

Class 3 is completely isolated (degree 0 in the anti-correlation graph).

The graph consists of two connected components, algebraic connectivity $\lambda_2 = 0$.

**IX. GRAM DIFFUSION AS AN ORNSTEIN-UHLENBECK PROCESS**
Restoring force: $\mu(\phi) \approx -0.985 \cdot \phi$ ($R^2 = 0.998$).

Phase volatility: $\sigma_\phi \approx 0.304$.

Stationary phase distribution is Gaussian with $\sigma_\infty^2 = \sigma_\phi^2 / (2\gamma) \approx 0.047$.

Correlation of theory with empirical distribution of $\Delta$: $r = 0.9985$.

**X. CONTINUOUS LIMIT: FOKKER-PLANCK EQUATION**
Discrete Langevin equation: $\Delta m_n = \mu(\text{PF}_n) + \sigma \xi_n$, where $\sigma = 0.596$.

Drift $\mu(\text{PF})$ is sigmoidal: $R^2 = 0.9659$.

Langevin simulation reproduces the empirical distribution of $\Delta m$: KS-test $p = 0.195$.

Fokker-Planck equation for phase density $\rho(\phi, \tau)$:
∂ρ/∂τ = γ ∂/∂ϕ (ϕρ) + (σ_ϕ²/2) ∂²ρ/∂ϕ².

**XI. HYBRID OPERATOR**
Hybrid Hamiltonian: $H = H_{12} + V_{\text{Prime}}$ (diagonal Prime Field potential).

Nonlinear scaling: $\gamma_n \approx a \lambda_n + b + c \ln(\lambda_n + 1)$ — accounts for zero density growth.

Calibrated on 6 zeros, tested on the next 6: correlation $r > 0.95$, average error $\sim 0.5$.

**XII. LEHMER PAIRS AND ADDITIONAL CHECKS**
Lehmer pairs (close zeros) have an anomalous Prime Field ($p < 0.001$).

Stall frequency for Lehmer pairs is 2.5 times higher than average.

L-function mod 101: on 25 zeros ($t \sim 30$) $k \approx 0.82$ — connections are stronger at low heights.

Class autocorrelation at lags multiple of 12: $r \approx 0.79$ — confirmation of $\mathbb{Z}_{12}$-periodicity.

**XIII. CONNECTION TO THE RIEMANN HYPOTHESIS**
The critical line is an attractor of the dynamics: restoring force $\mu(\phi) \propto -\phi$ always pushes phase toward $\phi = 0$.

Harmonic potential $V(\phi) = \frac{1}{2}\gamma \phi^2$ has a minimum on the critical line.

Proof program for RH:

1. Derive $\mu(\phi)$ analytically from the explicit formula.
2. Prove existence and uniqueness of a stationary solution to the Fokker-Planck equation.
3. Show that the spectrum of the linearized operator coincides with ${\gamma_n}$.

**XIV. SUMMARY**
| Category | Key Result |
| :--- | :--- |
| Structure | 12 streams, non-uniformity $p = 5.4 \times 10^{-7}$ |
| Anomaly | Bound pair (6,7), Class 3 isolated |
| Limit | $k_\infty = 12/10 = 1.200$ (error $0.5\%$ at $t \sim 10^{11}$) |
| Control | Prime Field controls Gram diffusion ($p < 10^{-300}$) |
| Operator | $H$ correlates with zeros: $r = 0.9275 \pm 0.0135$ (2,000 blocks) |
| Dynamics | Ornstein-Uhlenbeck process + Fokker-Planck equation |
| Invariance | $\text{Tr}(M^p) \propto p$ — exact scale symmetry |
| Universality | 100% of 12-zero blocks have $r > 0.85$ |
