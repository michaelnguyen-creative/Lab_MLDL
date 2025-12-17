# Project 1: Order Value Prediction — Requirements Document

**Project Code**: `P1-ORDER-VALUE`  
**Type**: Supervised Learning — Regression  
**Status**: Requirements Definition  
**Created**: 2024-12-17  
**Owner**: [Your Name]

---

## 📋 Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Target Variable Specification](#2-target-variable-specification)
3. [Feature Scope & Leakage Policy](#3-feature-scope--leakage-policy)
4. [Data Split Strategy](#4-data-split-strategy)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Success Criteria](#6-success-criteria)
7. [Experiment Plan](#7-experiment-plan)
8. [Risk Assessment](#8-risk-assessment)
9. [Deliverables Checklist](#9-deliverables-checklist)

---

## 1. Problem Definition

### 1.1 Business Context

**Scenario**:  
AdventureWorks' sales operations team wants to understand what drives order value variation. Specifically:
- Can we predict order size from customer/product/contextual signals?
- Which features have the strongest relationship with order value?
- Is there predictable structure, or is order value essentially random?

**Non-Goal**:  
This is **not** an operational forecasting system. We cannot predict order value "before" the order exists in a real-world sense, because in AdventureWorks, order headers and details are created simultaneously.

**Learning Objective**:  
> *"Given post-hoc features about an order (customer history, temporal context, product signals), can we recover the order total? This tests feature-target relationships and model expressiveness."*

### 1.2 Problem Statement

**Formal Definition**:
> Predict the total value of a sales order (`SalesOrderHeader.SubTotal`) using features derived from:
> - Customer historical behavior (pre-order)
> - Product catalog information
> - Temporal context
> - Geographic/territorial data
> - Salesperson attributes

**Prediction Grain**:
- **One row = one order** (`SalesOrderHeader` level)
- **Not** line-item level (though we may aggregate from `SalesOrderDetail`)

### 1.3 Assumptions & Constraints

**Assumptions**:
1. AdventureWorks data is **complete and accurate** (no missing critical fields)
2. Order values are **genuine transactions** (no test/void orders at scale)
3. Historical patterns are **somewhat stable** over the dataset timeframe
4. Customer IDs are **stable** (no ID changes mid-history)

**Constraints**:
1. **No future leakage**: Features must be computable from data available **before or at** order time
2. **Single-source features**: Only AdventureWorks tables (no external data)
3. **Computational**: Must train on standard Kaggle notebook hardware
4. **Interpretability preference**: If two models perform similarly, prefer the simpler/more interpretable one

---

## 2. Target Variable Specification

### 2.1 Primary Target

**Selected Target**: `SalesOrderHeader.SubTotal`

**Definition**:
- Sum of line item totals (Quantity × UnitPrice) **before** tax, freight, and discounts
- **Type**: Continuous, positive real number
- **Unit**: USD (assumed from AdventureWorks schema)

**Rationale**:
- ✅ Direct reflection of product value (sum of items purchased)
- ✅ Less influenced by external factors (tax rates, shipping method)
- ✅ Cleaner signal-to-noise for learning
- ❌ Does not capture full revenue impact (excludes freight, which can be significant)

### 2.2 Alternative Target (For Comparison)

**Alternative**: `SalesOrderHeader.TotalDue`

**Definition**:
- SubTotal + TaxAmt + Freight
- Full amount customer pays

**When to test**:
- After primary model is established
- To see if models can learn tax/freight patterns
- To compare feature importance shifts

### 2.3 Target Characteristics to Document

Before modeling, analyze and document:
- [ ] **Distribution**: Mean, median, std dev, skewness, kurtosis
- [ ] **Range**: Min, max, outliers (>3 std dev)
- [ ] **Nulls**: Any NULL SubTotal values? (should be none)
- [ ] **Zeros**: Any zero-value orders? (refunds? errors?)
- [ ] **Temporal drift**: Does distribution shift over time?
- [ ] **Log transform need**: If right-skewed, consider log(SubTotal) as target

**Decision Point**:  
> Should we predict **SubTotal** or **log(SubTotal)**?  
> → **Decide after Phase 1 data exploration**

---

## 3. Feature Scope & Leakage Policy

### 3.1 Feature Categories (Allowed)

#### **Category A: Customer Historical Features**
*Derived from orders placed **before** current order*

| Feature | Source | Leakage Risk |
|---------|--------|--------------|
| Customer lifetime order count | `SalesOrderHeader` (historic) | ✅ Safe |
| Customer average order value | `SalesOrderHeader.SubTotal` (historic) | ✅ Safe |
| Days since last order | `SalesOrderHeader.OrderDate` | ✅ Safe |
| Customer lifetime revenue | Sum of historic SubTotals | ✅ Safe |
| Customer order frequency | Orders per month (historic) | ✅ Safe |
| First order date | Earliest `OrderDate` | ✅ Safe |

**Note**: "Historic" = orders with `OrderDate < current order's OrderDate`

#### **Category B: Temporal Features**
*Extracted from current order's timestamp*

| Feature | Source | Leakage Risk |
|---------|--------|--------------|
| Order year | `SalesOrderHeader.OrderDate` | ✅ Safe |
| Order month | `SalesOrderHeader.OrderDate` | ✅ Safe |
| Order day of week | `SalesOrderHeader.OrderDate` | ✅ Safe |
| Order quarter | Derived from month | ✅ Safe |
| Is weekend | Derived from day of week | ✅ Safe |
| Days since dataset start | Monotonic time feature | ✅ Safe |

#### **Category C: Product Catalog Features**
*From dimension tables, assumed static*

| Feature | Source | Leakage Risk |
|---------|--------|--------------|
| Product list price (avg) | `Product.ListPrice` | ⚠️ Moderate (if prices change) |
| Product category | `ProductCategory.Name` | ✅ Safe |
| Product subcategory | `ProductSubcategory.Name` | ✅ Safe |
| Number of distinct products (historic customer avg) | Derived | ✅ Safe |

**Price handling**: If `Product.ListPrice` has temporal changes, we need price "as of order date". For simplicity in P1, assume **static prices** (document this limitation).

#### **Category D: Geographic/Territorial**

| Feature | Source | Leakage Risk |
|---------|--------|--------------|
| Territory ID | `SalesOrderHeader.TerritoryID` | ✅ Safe |
| Territory group | `SalesTerritory.Group` | ✅ Safe |
| Ship-to country | `Address.CountryRegionCode` (via joins) | ✅ Safe |

#### **Category E: Salesperson Features**

| Feature | Source | Leakage Risk |
|---------|--------|--------------|
| Salesperson ID | `SalesOrderHeader.SalesPersonID` | ✅ Safe |
| Salesperson tenure | `Employee.HireDate` | ✅ Safe |
| Salesperson historic avg order value | Derived from historic orders | ✅ Safe |

### 3.2 Feature Categories (FORBIDDEN — Leakage)

#### **Direct Order Contents**
❌ `SalesOrderDetail.OrderQty` — defines the target  
❌ `SalesOrderDetail.LineTotal` — is the target  
❌ `SalesOrderDetail.UnitPrice` — determines target  
❌ Count of line items **in current order** — derived from target  
❌ Distinct products **in current order** — reveals order composition

**Why forbidden**:  
These features are **consequences** of the target, not **predictors**. They exist simultaneously with SubTotal and cannot be known "before" the order value is determined.

**Exception (if explicitly testing)**:  
We **could** test: *"If we know the customer is buying N items, can we predict order value?"*  
This would be a **separate experiment** with explicit leakage acknowledgment.

#### **Future Information**
❌ `SalesOrderHeader.Status` — may change post-order  
❌ `SalesOrderHeader.ShipDate` — occurs after order  
❌ Any aggregates from orders **after** current order

### 3.3 Leakage Audit Checklist

Before using any feature, verify:
- [ ] **Temporal**: Is this computable from data **before** current order date?
- [ ] **Causal**: Does this **cause** the target, or is it **caused by** the target?
- [ ] **Availability**: Would this exist in a real-time prediction scenario?
- [ ] **Independence**: Is this derived from other orders in the test set?

**Document**: Any borderline features with justification for inclusion/exclusion

---

## 4. Data Split Strategy

### 4.1 Split Method Selection

**Three options considered**:

| **Method** | **Pros** | **Cons** | **Verdict** |
|------------|----------|----------|-------------|
| **Random 80/10/10** | Simple, maximizes training data | Ignores time, risks temporal leakage | ❌ Not preferred |
| **Temporal split** | Realistic (train on past, test on future) | Requires sufficient data in each period | ✅ **Recommended** |
| **Stratified by customer** | Ensures customer distribution | Complex, may not reflect deployment | ⚠️ Consider for comparison |

**Decision**: **Temporal split** (train on older orders, test on newer orders)

### 4.2 Temporal Split Specification

**Approach**:
1. Sort all orders by `OrderDate`
2. Define cutoff dates:
   - **Training set**: Orders from [earliest date] to [T1]
   - **Validation set**: Orders from (T1] to [T2]
   - **Test set**: Orders from (T2] to [latest date]

**Split percentages** (approximate):
- Train: 70% of temporal range
- Validation: 15% of temporal range
- Test: 15% of temporal range

**To be determined after Phase 1**:
- [ ] Exact date ranges (depends on data distribution)
- [ ] Whether to use calendar dates or order count percentiles
- [ ] Minimum order count per split (ensure statistical validity)

**Critical rule**:  
> **No customer history from validation/test sets may leak into training features.**

Example: If computing "customer lifetime orders" for a validation set order, only count orders **up to the training set cutoff**, not orders between training and validation cutoffs.

### 4.3 Alternative: Stratified Random (For Comparison)

**If temporal split shows severe distribution shift**, test:
- Random split stratified by:
  - Customer type (new vs. repeat)
  - Order size bucket (small/medium/large)
  - Territory

**Purpose**: Isolate whether poor test performance is due to temporal drift vs. model inadequacy

---

## 5. Evaluation Framework

### 5.1 Primary Metrics

**Primary Metric**: **RMSE (Root Mean Squared Error)**

**Definition**:
```
RMSE = sqrt(mean((y_pred - y_true)²))
```

**Why RMSE**:
- ✅ Penalizes large errors heavily (important for business impact)
- ✅ Same unit as target (USD)
- ✅ Differentiable (useful for neural optimization)
- ❌ Sensitive to outliers (document outlier handling strategy)

**Interpretation**:  
"On average, predictions are off by $X" (though squared errors mean it's pessimistic)

### 5.2 Secondary Metrics

**Metric 2**: **MAE (Mean Absolute Error)**

**Definition**:
```
MAE = mean(|y_pred - y_true|)
```

**Why MAE**:
- ✅ More interpretable than RMSE ("average dollar error")
- ✅ Robust to outliers
- ✅ Aligns with business thinking (absolute error magnitude)

**Metric 3**: **R² (Coefficient of Determination)**

**Definition**:
```
R² = 1 - (SS_residual / SS_total)
```

**Why R²**:
- ✅ Normalized (0-1 scale, or negative if worse than mean)
- ✅ Shows "variance explained"
- ✅ Model comparison across different target scales

**Interpretation**:  
- R² = 0.85 → "Model explains 85% of variance in order values"
- R² < 0 → "Model is worse than predicting the mean"

### 5.3 Additional Diagnostic Metrics

- **MAPE** (Mean Absolute Percentage Error): Only if no zero targets
- **Median Absolute Error**: Robustness check
- **Max Error**: Identify worst-case predictions
- **Residual plots**: Visual assessment of bias patterns

### 5.4 Evaluation Protocol

**For each model**:
1. Train on training set
2. Tune hyperparameters on validation set (if applicable)
3. Report final metrics on **test set only** (no peeking during development)
4. Document:
   - Training time
   - Inference time (per 1000 predictions)
   - Memory footprint (if significant)

**Statistical validity** (if time permits):
- Run 5-fold CV on training set for model comparison
- Report mean ± std dev for metrics
- Use validation set only for final architecture decisions

---

## 6. Success Criteria

### 6.1 Minimum Viable Success (Phase 2 — Baseline)

**Goal**: Establish performance floor

**Criteria**:
- [ ] **Naive baseline documented**: Mean prediction RMSE calculated
- [ ] **Customer-based heuristic tested**: "Predict customer's historical average order value"
- [ ] **Performance floor**: If R² > 0.80 with simple mean, task may be too easy

**Threshold**:  
Baseline R² should be **< 0.70** to justify further modeling (else signal is trivial)

### 6.2 ML Success (Phase 4)

**Goal**: Beat baseline with interpretable ML

**Criteria**:
- [ ] **Linear regression**: R² improvement over baseline by ≥0.05
- [ ] **GBDT (XGBoost/LightGBM)**: R² improvement over linear by ≥0.03
- [ ] **Feature importance**: Top 5 features explain ≥60% of importance
- [ ] **Error analysis**: Residuals show no systematic bias patterns

**Target**:  
- RMSE ≤ $500 (to be adjusted after seeing baseline)
- R² ≥ 0.75 (or justify why lower is acceptable)

### 6.3 DL Exploration (Phase 5)

**Goal**: Determine if neural methods add value

**Criteria**:
- [ ] **Justification documented**: Why we think DL might help (e.g., "high-cardinality categoricals benefit from embeddings")
- [ ] **Fair comparison**: Same features, same train/val/test split, comparable hyperparameter tuning effort
- [ ] **Performance**: DL must beat GBDT by ≥2% RMSE to be "worth it"
- [ ] **Interpretation**: If DL wins, explain **what** it learned that GBDT couldn't

**Honesty check**:  
If DL **does not** outperform GBDT, document:
- Why we expected it might
- What the comparison revealed
- Under what conditions DL might be worth trying

### 6.4 Learning Success (Phase 6 — Reflection)

**Goal**: Extract methodological insights

**Questions to answer**:
- [ ] **Model selection**: Which model would you deploy, and why? (not just "best metric")
- [ ] **Feature drivers**: What actually predicts order value?
- [ ] **Failure modes**: Where do all models struggle? (e.g., new customers, outlier products)
- [ ] **DL value**: Was the complexity justified?
- [ ] **Generalization**: What assumptions rely on AdventureWorks being clean?

**Deliverable**: 500-1000 word reflection in final notebook

---

## 7. Experiment Plan

### 7.1 Phase Sequence

| **Phase** | **Deliverable** | **Duration (Est.)** | **Go/No-Go Decision** |
|-----------|-----------------|---------------------|------------------------|
| **Phase 0** | This requirements doc | N/A | ✅ Complete |
| **Phase 1** | Data exploration notebook | 1-2 days | Target distribution acceptable? |
| **Phase 2** | Baseline performance | 0.5 day | Is task non-trivial (R² < 0.70)? |
| **Phase 3** | Feature engineering | 1-2 days | Leakage audit passed? |
| **Phase 4** | ML model comparison | 1-2 days | Plateau reached? |
| **Phase 5** | DL experiments (optional) | 1-2 days | Justified? Performance gain? |
| **Phase 6** | Critical reflection | 0.5 day | Insights documented? |

**Total estimated time**: 5-8 days (flexible based on findings)

### 7.2 Experiment Tracking

**Format**: Markdown table in notebook or separate `experiments.md`

**Template**:
```markdown
| Exp ID | Model | Features | Train RMSE | Val RMSE | Test RMSE | R² | Notes |
|--------|-------|----------|------------|----------|-----------|----|----|
| E001 | Mean baseline | N/A | X | X | X | X | Naive predictor |
| E002 | Customer avg | 1 feature | X | X | X | X | Historical heuristic |
| E003 | Linear (5 feat) | Customer + Time | X | X | X | X | Baseline ML |
| ... | ... | ... | ... | ... | ... | ... | ... |
```

**Mandatory columns**:
- Experiment ID (sequential)
- Model type
- Feature set (link to feature list)
- Train/Val/Test metrics (at minimum, RMSE and R²)
- Notes (hypotheses, observations, next steps)

---

## 8. Risk Assessment

### 8.1 Data Risks

| **Risk** | **Likelihood** | **Impact** | **Mitigation** |
|----------|----------------|------------|----------------|
| Target has outliers | High | Medium | Document outliers, test with/without, consider log transform |
| Temporal distribution shift | Medium | High | Use temporal split, analyze shift magnitude, consider rolling window |
| Customer history sparse | Medium | Medium | Segment by "new" vs "repeat", engineer time-since-first-order |
| Missing SalesPersonID | Low | Low | Create "unknown" category, check if systematic |
| Product price changes | Low | Medium | Assume static for P1, document limitation |

### 8.2 Methodological Risks

| **Risk** | **Likelihood** | **Impact** | **Mitigation** |
|----------|----------------|------------|----------------|
| Leakage not caught | Medium | **Critical** | Triple-check feature definitions, peer review |
| Task too easy (R² > 0.90) | Low | High | Switch to harder target (e.g., per-item price) |
| Task too hard (R² < 0.30) | Low | Medium | Check for data quality issues, re-examine target |
| Overfitting on validation set | Medium | High | Use test set sparingly, prefer CV on train |
| DL experiment becomes time sink | Medium | Low | Set time budget, have "kill criteria" |

### 8.3 Learning Risks

| **Risk** | **Likelihood** | **Impact** | **Mitigation** |
|----------|----------------|------------|----------------|
| Focus on metrics over understanding | High | Medium | Require Phase 6 reflection, prioritize error analysis |
| Skip baselines | Medium | High | Make Phase 2 mandatory go/no-go gate |
| Confirmation bias (force DL to "win") | Medium | Medium | Pre-commit to reporting honest results |
| Scope creep (add too many features) | Medium | Low | Start with minimal feature set, iterate |

---

## 9. Deliverables Checklist

### 9.1 Code Artifacts

- [ ] **01_data_exploration.ipynb**: Phase 1 analysis
  - Target distribution plots
  - Feature summary statistics
  - Temporal trend analysis
  - Null/missing value report
  
- [ ] **02_baseline_models.ipynb**: Phase 2 baselines
  - Mean/median prediction
  - Customer historical average
  - Simple linear regression
  - Baseline comparison table
  
- [ ] **03_feature_engineering.ipynb**: Phase 3 feature creation
  - Feature definitions (with leakage justification)
  - Feature distribution analysis
  - Correlation matrix
  - Leakage audit results
  
- [ ] **04_ml_models.ipynb**: Phase 4 ML experiments
  - Linear regression (Ridge/Lasso)
  - Random Forest
  - XGBoost/LightGBM
  - Model comparison table
  - Feature importance analysis
  
- [ ] **05_dl_experiments.ipynb** (optional): Phase 5 DL
  - Feedforward NN (with embeddings)
  - Fair comparison vs. GBDT
  - Architecture justification
  - Performance analysis
  
- [ ] **06_error_analysis.ipynb**: Diagnostic deep-dive
  - Residual plots
  - Error distribution by segment
  - Worst predictions investigation
  - Model failure modes

### 9.2 Documentation Artifacts

- [ ] **requirements.md** (this document)
- [ ] **experiments.md**: Experiment log table
- [ ] **reflection.md**: Phase 6 critical analysis
  - What worked / didn't work
  - ML vs DL assessment
  - Generalization concerns
  - What I learned
  
- [ ] **README.md** (project-level): Summary for external viewers
  - Problem statement
  - Key findings (best model, RMSE, R²)
  - Feature importance summary
  - Limitations and caveats

### 9.3 Visualization Artifacts (Minimal)

- [ ] Target distribution histogram
- [ ] Residual plot (actual vs. predicted)
- [ ] Feature importance bar chart (top 10)
- [ ] Learning curves (train vs. val loss over epochs, if DL)

**Note**: Prefer tables and text over complex visualizations

---

## 10. Open Questions (To Resolve in Phase 1)

Before proceeding to implementation, answer:

1. **Target transformation**:
   - [ ] Should we predict `SubTotal` or `log(SubTotal)`?
   - [ ] Decision criterion: If right-skewed (skewness > 1), use log

2. **Temporal split cutoffs**:
   - [ ] What are the exact train/val/test date ranges?
   - [ ] Do we have sufficient orders in each split? (min 10K per split recommended)

3. **Customer history definition**:
   - [ ] For customers with no history, use global average or separate "new customer" model?
   - [ ] How to handle customers with only 1 prior order?

4. **Outlier handling**:
   - [ ] Define outlier threshold (e.g., >3 std dev, or >99th percentile)
   - [ ] Should we cap/remove outliers, or keep for model robustness?

5. **Feature engineering priorities**:
   - [ ] Start with 5-10 core features or engineer 50+ features?
   - [ ] Recommendation: Start minimal, add incrementally

6. **DL experiment trigger**:
   - [ ] If GBDT R² > 0.85, is DL still worth trying? (Answer: only if learning-focused)
   - [ ] If GBDT R² < 0.60, fix features before trying DL

---

## 11. Approval & Sign-Off

**Requirements Status**: ✅ **APPROVED** (self-approved, solo learning)

**Next Action**: Proceed to Phase 1 (Data Exploration)

**Expected Start**: [Date you begin Phase 1]

**Notes for Future Self**:
- Remember: Baselines are mandatory, not optional
- If you skip leakage audit, you're lying to yourself
- Negative results are valid results
- Document assumptions as you make them, not after the fact

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-17  
**Status**: Ready for Phase 1

---

## 🚦 Decision Summary (Quick Reference)

| **Decision Point** | **Choice** | **Rationale** |
|--------------------|------------|---------------|
| **Target variable** | `SubTotal` | Cleaner signal than `TotalDue` |
| **Target transform** | TBD (Phase 1) | Depends on distribution |
| **Split strategy** | Temporal 70/15/15 | Realistic, prevents temporal leakage |
| **Primary metric** | RMSE | Penalizes large errors |
| **Secondary metrics** | MAE, R² | Interpretability + variance explained |
| **Baseline threshold** | R² < 0.70 | Ensures task is non-trivial |
| **DL trigger** | GBDT R² < 0.85 | Only if complexity justified |
| **Feature policy** | Minimal → incremental | Avoid overfitting to AdventureWorks |
