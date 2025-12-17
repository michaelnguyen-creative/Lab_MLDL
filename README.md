# 🧪 ML/DL Methodology Lab — AdventureWorks

> **A structured learning environment for rigorous machine learning experimentation, critical analysis, and methodological development using enterprise-grade data.**

---

## 📋 Lab Overview

### Purpose

This lab is a **controlled learning environment** for developing, testing, and critically evaluating machine learning and deep learning methodologies on structured, enterprise-style data. 

**This is NOT**:
- A production ML system
- A portfolio of "best practices" implementations
- An attempt to achieve state-of-the-art performance
- A collection of tutorial reproductions

**This IS**:
- A methodological sandbox for **learning through experimentation**
- A framework for **comparing approaches** with empirical rigor
- A practice ground for **critical thinking** about when/why ML/DL methods work
- Documentation of **honest learning** — including failures, dead ends, and "it depends" conclusions

---

## 🎯 Core Learning Objectives

### 1. **Methodological Rigor**
- Establish proper baselines before complex models
- Design fair comparisons (same data, same metrics, same splits)
- Distinguish between **model performance** and **engineering quality**
- Recognize when simplicity outperforms complexity

### 2. **Critical Model Evaluation**
- Question whether ML is needed at all
- Identify when DL justifies its cost vs. classical ML
- Understand **where models fail** (error analysis over accuracy chasing)
- Detect data leakage, overfitting, and spurious patterns

### 3. **Structured Experimentation**
- Document assumptions explicitly
- Track experiments systematically
- Version control methodology (not just code)
- Write retrospectives on what worked/didn't

### 4. **Tabular Data Expertise**
- Master feature engineering for structured data
- Understand tree-based vs. neural approaches
- Handle temporal structure, categorical encodings, and imbalance
- Recognize dataset-specific vs. generalizable insights

---

## 🗂️ Lab Structure

### Project Organization

Each project follows a **phase-gated methodology**:

```
Phase 0: Problem Framing
├── Business context definition
├── Target variable justification  
├── Success criteria specification
└── Leakage audit

Phase 1: Data Understanding
├── Target distribution analysis
├── Feature availability mapping
├── Temporal/split strategy design
└── Baseline definition

Phase 2: Baseline Establishment
├── Naive predictors (mean, mode, etc.)
├── Simple heuristics
├── Performance floor documentation
└── "Is ML needed?" checkpoint

Phase 3: Feature Engineering
├── Domain-driven features
├── Aggregations and derivations
├── Encoding strategies
└── Leakage re-audit

Phase 4: ML Model Comparison
├── Linear models (interpretability baseline)
├── Tree ensembles (GBDT standard)
├── Model selection criteria
└── Performance plateau identification

Phase 5: DL Experimentation (conditional)
├── Justification for neural methods
├── Architecture selection rationale
├── Fair comparison vs. ML
└── Cost-benefit analysis

Phase 6: Critical Reflection
├── Error pattern analysis
├── Model selection justification
├── Generalization concerns
├── "What did I actually learn?"
```

### Project Portfolio (Planned)

| **Project** | **Type** | **Complexity** | **DL Justification** | **Status** |
|-------------|----------|----------------|----------------------|------------|
| **P1: Order Value Prediction** | Regression | Low | Weak (learning exercise) | 🔄 In Progress |
| **P2: Customer Churn** | Classification | Medium | Weak (imbalanced data study) | 📋 Planned |
| **P3: Product Recommendation** | Ranking/Collaborative Filtering | High | Moderate (embeddings) | 📋 Planned |
| **P4: Sales Forecasting** | Time Series | Medium | Weak (vs. classical methods) | 📋 Planned |
| **P5: Demand Prediction** | Multi-variate TS | High | Moderate (spatial-temporal) | 📋 Planned |

*(Projects may be reordered, merged, or replaced based on learning outcomes)*

---

## 🗃️ Dataset: AdventureWorks 2025

### Why AdventureWorks?

**Advantages**:
- ✅ **Enterprise-realistic structure**: Proper normalization, foreign keys, business logic
- ✅ **Rich feature space**: Customer, Product, Employee, Territory, Time dimensions
- ✅ **Clean and documented**: Focus on methodology, not data wrangling
- ✅ **Pedagogically complete**: Supports diverse ML tasks (regression, classification, time-series, recommendations)
- ✅ **Reproducible**: Public dataset with stable schema

**Critical Limitation**:
- ⚠️ **Not representative of real-world messiness**: 
  - Minimal missing data
  - No adversarial actors
  - Static schema
  - Pre-cleaned outliers
  
**Implication**: Results demonstrate **methodology**, not production readiness. Generalization claims must acknowledge this.

### Dataset Scope

**Tables in use**:
- **Sales**: `SalesOrderHeader`, `SalesOrderDetail`, `Customer`, `SalesTerritory`
- **Product**: `Product`, `ProductCategory`, `ProductSubcategory`
- **Person**: `Person`, `Address`, `BusinessEntity`
- **HumanResources**: `Employee`
- **Purchasing**: (as needed for specific projects)

**Data characteristics** (from prior analysis):
- ~500K sales orders (2021-2024)
- ~60K customers
- ~28K products
- Temporal range suitable for train/test splits

---

## 🧭 Guiding Principles

### 1. **Baseline-First Mentality**
> *"Never use a complex model without knowing what a simple one achieves."*

Every project **must** document:
- Naive baseline (e.g., mean prediction, majority class)
- Simple heuristic (e.g., customer historical average)
- Linear/logistic regression baseline

### 2. **Honest Reporting**
> *"Negative results are results."*

Document:
- What **didn't** work and why
- When DL **underperformed** simpler methods
- Experiments that were abandoned (with reasons)

### 3. **Explicitness Over Assumed Knowledge**
> *"If it's not written down, it didn't happen."*

Every experiment includes:
- Clear problem statement
- Exact target definition
- Train/val/test split strategy
- Evaluation metric justification
- Feature leakage audit

### 4. **Critique Over Celebration**
> *"Question the model, not just the accuracy number."*

Always ask:
- **Where does this model fail?** (error analysis)
- **What assumptions does this rely on?** (data, distribution, causality)
- **What would break this in production?** (schema drift, distribution shift, adversarial inputs)
- **Is this learning signal or noise?** (especially on clean data)

### 5. **DL Is Not Default**
> *"Use deep learning when you can justify it, not because it's fashionable."*

DL experiments require:
- Documented reason why classical ML is insufficient
- Hypothesis about what DL might learn that ML can't
- Fair comparison with proper hyperparameter tuning on both sides
- Honest assessment of whether added complexity paid off

---

## 📊 Evaluation Standards

### Model Comparison Requirements

All model comparisons must use:
- ✅ **Identical datasets** (same train/val/test split)
- ✅ **Identical features** (unless feature learning is the comparison point)
- ✅ **Identical metrics** (primary + secondary for context)
- ✅ **Statistical validation** (multiple runs if stochastic, confidence intervals where applicable)
- ✅ **Computational cost tracking** (training time, inference time, memory)

### Deliverable Checklist (Per Project)

Every completed project must include:

**Empirical Artifacts**:
- [ ] Baseline performance table
- [ ] Model comparison table (≥3 models)
- [ ] Feature importance analysis
- [ ] Error distribution analysis
- [ ] Learning curves (if applicable)

**Methodological Documentation**:
- [ ] Problem statement with business context
- [ ] Target variable justification
- [ ] Train/val/test split rationale
- [ ] Leakage audit results
- [ ] Metric selection justification

**Critical Analysis**:
- [ ] Error pattern discussion
- [ ] Model selection justification (not just "best RMSE")
- [ ] Failure mode identification
- [ ] Generalization limitations
- [ ] "What I learned" reflection

---

## 🛠️ Technical Stack

### Primary Platform
- **Kaggle Notebooks**: Experiment hosting, version control, dataset storage
  - Why: Reproducibility, GPU access, public sharing capability

### Tools (Flexible)
- **Data**: pandas, NumPy, SQL (for AdventureWorks queries)
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **DL**: PyTorch (preferred) or TensorFlow/Keras
- **Evaluation**: scikit-learn metrics, custom evaluation scripts
- **Visualization**: matplotlib, seaborn (minimal, text-first reporting)

### Version Control
- **Kaggle notebook versions** for experiment tracking
- **GitHub** (optional): For external documentation, markdown reports, cross-project analysis

---

## 📐 Documentation Standards

### Notebook Structure (Standard Template)

```markdown
# [Project Name]: [Specific Task]

## 1. Problem Framing
- Business context
- Target definition
- Success criteria

## 2. Data Understanding
- Target distribution
- Feature summary
- Split strategy

## 3. Baseline
- Naive predictor
- Simple heuristic
- Performance floor

## 4. Feature Engineering
- Feature categories
- Leakage audit
- Encoding decisions

## 5. Model Experiments
- Model comparison table
- Hyperparameter notes
- Training observations

## 6. Error Analysis
- Residual plots
- Failure patterns
- Edge cases

## 7. Critical Reflection
- What worked / didn't
- DL vs ML assessment
- Generalization concerns
- Next steps
```

### Terminology Standards

To avoid ambiguity:
- **OLTP**: Operational database (AdventureWorks source)
- **Target**: Dependent variable (y)
- **Features**: Independent variables (X)
- **Baseline**: Non-ML reference point
- **Leakage**: Information unavailable at prediction time
- **Grain**: Unit of prediction (e.g., one row = one order)

---

## 🚀 Current Status

### Active Project
**P1: Order Value Prediction** (Regression)
- 🎯 Objective: Predict `SalesOrderHeader.SubTotal` from pre-order features
- 📍 Phase: Problem framing complete, awaiting detailed requirements
- 📂 Location: `/projects/01_order_value_prediction/`

### Next Steps
1. Finalize P1 requirements (target, split, features)
2. Complete P1 Phases 1-6
3. Document P1 learnings
4. Review lab methodology based on P1 experience
5. Define P2 based on identified knowledge gaps

---

## 🧠 Meta-Learning Objectives

Beyond individual project skills, this lab aims to develop:

### Judgment
- When to use ML vs. rules/heuristics
- When DL is worth the complexity
- When to stop experimenting and ship

### Rigor
- Designing fair comparisons
- Avoiding confirmation bias
- Documenting negative results

### Communication
- Explaining models to non-technical stakeholders
- Justifying architectural choices
- Writing honest retrospectives

### Humility
- Recognizing dataset limitations
- Admitting when simpler methods win
- Distinguishing "works on AdventureWorks" from "works in production"

---

## 📝 Contributing to This Lab

This is a **solo learning environment**, but documentation follows open-source standards:

- **Issues/Questions**: Document in project-specific markdown
- **Methodology improvements**: Update this README
- **New projects**: Follow phase-gated template
- **External feedback**: Welcome via Kaggle comments or GitHub issues

---

## 📚 References & Inspiration

**Methodological**:
- Domingos, P. (2012). "A Few Useful Things to Know about Machine Learning"
- Provost & Fawcett (2013). "Data Science for Business"
- Goodfellow et al. (2016). "Deep Learning" (Chapter 5: ML Basics)

**Experimental Design**:
- Hulten, G. (2018). "Building Intelligent Systems"
- Raschka & Mirjalili (2019). "Python Machine Learning" (Chapter 6: Model Evaluation)

**Dataset**:
- Microsoft AdventureWorks 2025 Sample Database
- [Documentation](https://learn.microsoft.com/en-us/sql/samples/adventureworks-install-configure)

---

## 🔗 Quick Links

- **Kaggle Profile**: [Your Profile]
- **Project Index**: See `/projects/README.md`
- **Experiment Log**: See `/experiments/log.md`
- **Reflections**: See `/reflections/`

---

## 📊 Lab Metrics (Self-Assessment)

Track over time:
- Projects completed: **0 / 5**
- Baselines documented: **0 / 5**
- DL experiments justified: **0 / ?**
- Honest negative results: **0 / ?**
- "I was wrong about..." moments: **0 / ?**

**Last Updated**: 2025-12-17

---

*"The goal is not to build the best model. The goal is to understand when, why, and whether building models is the right choice at all."*
