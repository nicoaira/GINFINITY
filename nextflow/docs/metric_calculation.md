

## 1. Data selection via percentile 

 
- We begin with a full table of all possible windows $$j$$ comparing exon $$A$$ and exon $$B$$, each with a learned distance $$d_j$$.
 
- To focus on the strongest structural matches, we sort by $$d_j$$ ascending and keep only the top $$p$$% (percentile) of rows.  Denote the resulting set by indices $$j=1,\dots,N$$.



---



## 2. Window notation and ranking 

For each retained window $$j$$:
 
- **Positions** 
 
  - $$s^A_j,e^A_j$$: start/end base indices on exon $$A$$ (so the window covers bases $$a=s^A_j,\dots,e^A_j$$).

 
- **Rank**  $$r_j\in\{1,\dots,N\}$$: the sorted order by distance ($$r=1$$ is best/smallest $$d$$).



---


3. Per-cell matrix and the numerator $$N_{ab}$$
We build an $$L\times M$$ matrix (where $$L=|A|$$, $$M=|B|$$) that accumulates **all**  the contributions of every window that covers a given base-pair $$(a,b)$$.
 
2. **Diagonal offset** 
$$
   \Delta_{ab}^{(j)}
  = \bigl|\,(a - s^A_j)\;-\;(b - s^B_j)\bigr|
  \quad(\Delta\ge0). 
$$

This measures how far $$(a,b)$$ is from “perfectly aligned” within window $$j$$.
 
4. **Rank weight for signal** 
$$
   f_\mathrm{num}(r_j)
  = r_j^{-\alpha_1}, 
$$

so the best-ranked windows contribute more; tuning exponent $$\alpha_1$$ sharpens or flattens that drop-off.
 
6. **Positional (off-diagonal) weight** 
$$
   w_\mathrm{num}^{(j)}(a,b)
  = \exp\!\bigl(-\beta_1\,\Delta_{ab}^{(j)}\bigr), 
$$

so cells nearer the diagonal (where $$\Delta=0$$) contribute most; $$\beta_1$$ controls how rapidly off-diagonal signal decays.

Putting these together, the **numerator**  accumulates:
$$
 \boxed{
N_{ab}
\;=\;
\sum_{j:\,(a,b)\in\text{window}_j}
  f_\mathrm{num}(r_j)\;
  w_\mathrm{num}^{(j)}(a,b)
\;=\;
\sum_{j:\,(a,b)\in\text{window}_j}
  \frac{1}{r_j^{\alpha_1}}
  \exp\bigl(-\beta_1\,\Delta_{ab}^{(j)}\bigr).
} 
$$

 
- **Interpretation:** 
 
  - Every window $$j$$ that covers base-pair $$(a,b)$$ “votes” its strength into $$N_{ab}$$.
 
  - Votes from high-quality matches ($$r_j$$ small, $$\Delta$$ small) are weighted heavily.
 
  - This term **aggregates all signal**  that supports $$(a,b)$$ as structurally similar.



---


4. Denominator $$D_{ab}$$ as redundancy penalty
When many overlapping windows cover the same $$(a,b)$$, we risk **over-counting**  the same structural signal.  To counteract that, we build a parallel accumulator:
$$
 \boxed{
D_{ab}
\;=\;
\sum_{j:\,(a,b)\in\text{window}_j}
  f_\mathrm{den}(r_j)\;
  w_\mathrm{den}^{(j)}(a,b)
\;=\;
\sum_{j:\,(a,b)\in\text{window}_j}
  \frac{1}{r_j^{\alpha_2}}
  \exp\bigl(-\beta_2\,\Delta_{ab}^{(j)}\bigr).
} 
$$

 
- Here we allow separate exponents $$\alpha_2,\beta_2$$ so you can penalize overlap differently than you reward signal.
 
- **Interpretation:** 
 
  - $$D_{ab}$$ measures the **total “mass”**  of windows supporting $$(a,b)$$, again weighted by rank and diagonal proximity.
 
  - A large $$D_{ab}$$ indicates many (possibly redundant) windows covering that cell.



---



## 5. Combining signal and redundancy 

To balance aggregating signal with penalizing redundancy, we introduce exponent $$\gamma\in[0,1]$$:
$$
 \boxed{
M_{ab}
\;=\;
\frac{N_{ab}}
     {D_{ab}^{\,\gamma}}
\quad
\bigl(D_{ab}>0\bigr),
} 
$$

 
- If $$\gamma=0$$: no penalty, $$M_{ab}=N_{ab}$$.
 
- If $$\gamma=1$$: full normalization, $$M_{ab}=N_{ab}/D_{ab}$$ (an average‐style).
 
- Intermediate $$\gamma$$ gives **diminishing returns**  on added redundancy: each extra overlapping window boosts $$D_{ab}$$ and thus divides $$N_{ab}$$ by a larger factor.



---



## 6. Global score without size normalization 

Finally, we sum over **all**  base-pairs:
$$
 \boxed{
G(A,B)
=\sum_{a=1}^{L}\sum_{b=1}^{M}M_{ab}.
} 
$$

 
- **Why sum?**  Summation rewards both the **breadth**  of coverage (many interacting cells) and the **depth**  of strong signals, without shrinking scores for longer exons.



---



## 7. Hyperparameter tuning with Optuna 

We tune $$\{p,\alpha_1,\alpha_2,\beta_1,\beta_2,\gamma\}$$ to maximize the ability to **separate**  one known true‐positive exon pair

$$\bigl(\mathrm{ENSE00001655346.1},\mathrm{ENSE00004286647.1}\bigr)$$

from all others.
 
2. **Objective function** 
 
  - Compute $$G_{\rm tp}$$ for the true pair and collect all other scores $$G_{i}$$.
 
  - Define the **average of the top 3**  competitors:

$$\displaystyle \overline{G}_3 = \frac{1}{3}\sum_{k=1}^3 G_{(k)}$$,

where $$G_{(1)}\ge G_{(2)}\ge G_{(3)}$$.
 
  - Use the **relative margin** 
$$
   \text{margin}
  = \frac{G_{\rm tp}-\overline{G}_3}{\overline{G}_3}. 
$$

This ensures we measure **proportional separation** , not raw difference.
 
4. **Rank-1 enforcement** 

We require that once a trial places $$G_{\rm tp}$$ at the very top of the list (rank 1), only future trials that also do so are compared by margin.  This guarantees the selected hyperparameters actually make the true positive the **highest scoring**  pair.
 
6. **Bayesian search** 

Optuna’s sampler balances exploration and exploitation, rapidly focusing on promising regions of hyperparameter space.  We run multiple trials (e.g. 100–200) and then inspect:
 
  - **Optimization-history**  plot—how margin improves over trials.
 
  - **Parameter-importance**  plot—which hyperparameters most influence margin.
 
  - **Slice**  plots—margin as a function of each parameter.


These diagnostics help reviewers understand convergence and parameter sensitivity.

ChatGPT: 

Give it to me in Markdown format

You: 

# Methods 

We compare RNA secondary-structure similarity between two exons, $$A$$ and $$B$$, of lengths $$L=|A|$$ and $$M=|B|$$.  Below we describe each step of the metric’s construction, explain every symbol and parameter, and outline how we tune them.


---



## 1. Top-percentile filtering 

 
- **Input:**  a TSV of all window comparisons, each row $$j$$ with learned distance $$d_j$$.
 
- **Percentile**  $$p\in(0,100]$$: sort rows by increasing $$d_j$$ and keep only the first
$$
   N \;=\;\Bigl\lceil \tfrac{p}{100}\times(\text{total rows})\Bigr\rceil 
$$

windows, indexed $$j=1,\dots,N$$.
 
- **Rationale:**  focuses computation on the most-similar structural matches and reduces noise.



---



## 2. Window notation and ranking 

For each retained window $$j$$:
 
- $$(s^A_j,e^A_j)$$: start/end base indices on exon $$A$$ (covering bases $$a=s^A_j,\dots,e^A_j$$).
 
- $$(s^B_j,e^B_j)$$: start/end on exon $$B$$.
 
- **Rank**  $$r_j\in\{1,\dots,N\}$$: position in sorted order by $$d_j$$ (1 = best).



---


3. Per-cell matrix and **numerator**  $$N_{ab}$$
We build an $$L\times M$$ matrix that aggregates every window’s contribution to each base-pair $$(a,b)$$.
 
2. **Diagonal offset** 
$$
   \Delta_{ab}^{(j)}
  = \bigl|\,(a - s^A_j)\;-\;(b - s^B_j)\bigr|. 
$$

Zero if $$a$$ and $$b$$ align perfectly along the window diagonal.
 
4. **Rank-decay weight** 
$$
   f_\mathrm{num}(r_j)
  = r_j^{-\alpha_1},
  \quad \alpha_1>0 
$$

(steeper $$\alpha_1$$ → only top ranks contribute).
 
6. **Off-diagonal decay** 
$$
   w_\mathrm{num}^{(j)}(a,b)
  = \exp\!\bigl(-\beta_1\,\Delta_{ab}^{(j)}\bigr),
  \quad \beta_1\ge0 
$$

(larger $$\beta_1$$ → stronger focus on the diagonal).
 
8. **Numerator accumulation** 
$$
 \boxed{
N_{ab}
= \sum_{j:\,(a,b)\in\text{win}_j}
    \frac{1}{r_j^{\alpha_1}}
    \exp\!\bigl(-\beta_1\,\Delta_{ab}^{(j)}\bigr)
} 
$$

 
  - **Interpretation:**  sums all “votes” from windows covering $$(a,b)$$, weighted by rank and diagonal proximity.



---


4. **Denominator**  $$D_{ab}$$: redundancy penalty

To avoid over-counting overlapping windows:

 
2. **Rank-decay for penalty** 
$$
   f_\mathrm{den}(r_j)
  = r_j^{-\alpha_2},
  \quad \alpha_2>0 
$$
 
4. **Positional decay for penalty** 
$$
   w_\mathrm{den}^{(j)}(a,b)
  = \exp\!\bigl(-\beta_2\,\Delta_{ab}^{(j)}\bigr),
  \quad \beta_2\ge0 
$$
 
6. **Denominator accumulation** 
$$
 \boxed{
D_{ab}
= \sum_{j:\,(a,b)\in\text{win}_j}
    \frac{1}{r_j^{\alpha_2}}
    \exp\!\bigl(-\beta_2\,\Delta_{ab}^{(j)}\bigr)
} 
$$

 
  - **Interpretation:**  measures the total “mass” of overlapping windows—higher means more redundancy.



---



## 5. Combining signal and redundancy 

Introduce redundancy exponent $$\gamma\in[0,1]$$.  For each $$(a,b)$$ with $$D_{ab}>0$$, define:
$$
 \boxed{
M_{ab}
\;=\;
\frac{N_{ab}}{D_{ab}^{\,\gamma}}
}
\quad(\gamma=0\,\text{no penalty};\;\gamma=1\,\text{full normalization}). 
$$



---



## 6. Global similarity score 


Sum over all cells:

$$
 \boxed{
G(A,B)
=\sum_{a=1}^{L}\sum_{b=1}^{M} M_{ab}.
} 
$$

 
- **Why sum?** 
 
  - Rewards both **coverage**  (many interacting base-pairs) and **strength**  (high weights).
 
  - No normalization by $$L$$ or $$M$$ preserves magnitude differences as biological signal.



---



## 7. Hyperparameter optimization with Optuna 

We tune $$\{p,\alpha_1,\alpha_2,\beta_1,\beta_2,\gamma\}$$ using Optuna:
 
2. **Objective function** 
 
  - Compute $$G_{\rm tp}$$ for the known true-positive exon pair

$$(\mathrm{ENSE00001655346.1},\mathrm{ENSE00004286647.1})$$.
 
  - Let $$G_{(1)}\ge G_{(2)}\ge G_{(3)}$$ be the next three best scores.
 
  - Define the **relative margin**  vs. top-3 average:
$$
   \mathrm{margin}
  = \frac{G_{\rm tp}-\tfrac{1}{3}(G_{(1)}+G_{(2)}+G_{(3)})}
         {\tfrac{1}{3}(G_{(1)}+G_{(2)}+G_{(3)})}. 
$$
 
4. **Rank-1 enforcement** 
 
  - Once any trial ranks the true positive **first** , all subsequent trials that fail to do so are heavily penalized.
 
  - Ensures final parameters truly place the true positive at the top.
 
6. **Search strategy** 
 
  - Optuna uses a Bayesian sampler to efficiently explore the six-dimensional space.
 
  - We run ~100–200 trials, then inspect:

 
    - **Optimization-history** : margin vs. trial.
 
    - **Parameter-importance** : which parameters matter most.
 
    - **Slice plots** : margin vs. each parameter.


These diagnostics demonstrate convergence and parameter sensitivity for reviewer scrutiny.



---



### Table of parameters 

| Parameter | Symbol | Role | 
| --- | --- | --- | 
| Percentile | ppp | Fraction of top-distance rows kept | 
| Num-rank decay | α1\alpha_1α1​ | Exponent in numerator rank weight r−α1r^{-\alpha_1}r−α1​ | 
| Den-rank decay | α2\alpha_2α2​ | Exponent in denominator rank weight r−α2r^{-\alpha_2}r−α2​ | 
| Num-positional | β1\beta_1β1​ | Off-diagonal decay in numerator e−β1Δe^{-\beta_1\Delta}e−β1​Δ | 
| Den-positional | β2\beta_2β2​ | Off-diagonal decay in denominator e−β2Δe^{-\beta_2\Delta}e−β2​Δ | 
| Redundancy | γ\gammaγ | Exponent on DabD_{ab}Dab​ penalty (0=no penalty; 1=full normalization) | 

This framework balances **signal amplification**  (numerator) with **redundancy control**  (denominator), yielding a robust per-base and global similarity metric.


Formatear bien

https://chatgpt.com/c/68092171-014c-8006-be41-1f3ff1d74626