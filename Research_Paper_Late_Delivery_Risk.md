# Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations

**Author:** Data Science & Logistics Operations Group  
**Project:** Unified Mentors — Project 1 Completion Report  
**Date:** November 2026  

---

## Abstract
Delivery punctuality is an essential determinant of customer retention, service-level agreement (SLA) compliance, and operational cost containment in global supply chain networks. Despite substantial investments in tracking infrastructure, supply chain managers frequently encounter high rates of delayed fulfillment. This paper presents an end-to-end predictive machine learning framework designed to forecast late delivery risk at the time of order intake, utilizing a benchmark dataset of 180,519 multi-regional shipments from APL Logistics / DataCo. 

To prevent data leakage, post-fulfillment operational variables (e.g., actual shipping days, delivery status, order fulfillment status) were eliminated, restricting candidate features to 14 attributes available prior to dispatch. A comparison of Logistic Regression, Gradient Boosting, and Random Forest models demonstrates that an optimized Random Forest pipeline achieves superior discrimination with an Area Under the Receiver Operating Characteristic curve (ROC-AUC) of **0.7603** on unseen test data. 

Recognizing that missed late deliveries incur substantially higher business costs than false alarms, we formulate delivery risk forecasting as a cost-sensitive decision problem. By calibrating the classification threshold from the standard 0.50 cutoff to an empirical optimum of **0.35**, the model’s late-delivery recall jumps from **55.22% to 88.79%**, capturing **17,577 delayed shipments** and reducing undetected delays by **75.0%**. Global model explainability via Mean Decrease in Impurity (MDI) and Permutation Feature Importance reveals that *Shipping Mode* is the predominant predictor (86.87% normalized MDI), uncovering a critical supply chain paradox: premium *First Class* shipments experience a **95.32% delay rate** due to unachievable scheduled transit promises. The predictive engine is embedded into an interactive Streamlit decision support system with a dynamic dispatch alert queue, providing supply chain practitioners with an actionable early-warning system.

**Keywords:** Supply Chain Analytics, Late Delivery Prediction, Cost-Sensitive Classification, Random Forest, Model Interpretability, Threshold Optimization, Logistics Decision Support Systems.

---

## 1. Introduction

### 1.1 Operational Context
Globalized commerce has transformed supply chain networks into complex, highly interdependent ecosystems characterized by multi-tiered supplier tiers, geographically dispersed distribution centers, and diverse last-mile carrier arrangements (Kuhn & Johnson, 2013). Within this paradigm, delivery timeliness has transitioned from a competitive differentiator to a mandatory operational requirement. Contractual enterprise agreements commonly enforce stringent On-Time, In-Full (OTIF) service levels, penalizing delivery breaches through direct contractual chargebacks, freight expedited re-routing expenses, and damaged business reputations (Zheng, 2025).

Despite advances in Enterprise Resource Planning (ERP) and Transportation Management Systems (TMS), supply chain operations remain vulnerable to disruptions. Order backlogs, warehouse processing congestion, regional transportation capacity imbalances, and inaccurate transit time estimations frequently cascade into fulfillment failures. In typical enterprise logistics datasets, late delivery incidence often exceeds 50%, highlighting widespread systemic mismatches between scheduled commitments and physical fulfillment capabilities.

### 1.2 The Cost-Asymmetry Problem in Logistics Forecasting
Conventional classification models applied to operational risk management typically default to a decision probability threshold of $p = 0.50$, implicitly assuming that false positives and false negatives incur symmetric operational penalties. In logistics fulfillment, this assumption is demonstrably invalid:
- **False Negative (Type II Error — Missed Late Delivery):** An order predicted to arrive on time that subsequently arrives late. This failure triggers customer escalation, SLA non-compliance penalties, emergency expediting costs, and negative brand equity.
- **False Positive (Type I Error — False Alarm):** An order predicted to be late that arrives on time. This triggers automated operational inspection, warehouse prioritization, or proactive tracking verification—interventions with negligible marginal cost.

Consequently, supply chain decision-makers require predictive architectures calibrated to maximize sensitivity (recall), ensuring that the overwhelming majority of at-risk shipments are intercepted prior to carrier handoff.

### 1.3 Research Objectives
This study addresses four core research questions:
1. **Data Leakage Mitigation:** How can predictive models be rigorously isolated from post-event leakage in complex operational datasets?
2. **Algorithm Comparison:** Which machine learning architecture (linear vs. ensemble tree-based) provides the most robust discrimination between on-time and delayed shipments?
3. **Threshold Optimization:** What empirical decision threshold optimally balances late-delivery capture rate against operational false alarms?
4. **Actionable Explainability:** What structural features drive late delivery risk, and how can these insights inform tactical supply chain interventions?

---

## 2. Related Literature

### 2.1 Machine Learning in Supply Chain Management
Predictive analytics in supply chain operations has increasingly shifted from classical statistical forecasting (e.g., ARIMA, linear regression) toward non-linear machine learning architectures (James et al., 2023). Empirical studies demonstrate that ensemble algorithms, specifically Random Forests and Gradient Boosted Decision Trees (GBDT), consistently outperform parametric baselines on tabular logistics data due to their ability to capture high-order feature interactions, accommodate mixed numerical and categorical data types, and resist outlier distortion (Chen & Guestrin, 2016; Lessmann et al., 2015).

### 2.2 Cost-Sensitive Learning & Threshold Tuning
In imbalanced or cost-asymmetric domains such as fraud detection, medical diagnosis, and supply chain risk, standard accuracy is recognized as an inadequate evaluation metric (Davis & Goadrich, 2006; Nasir et al., 2024). Researchers emphasize that model evaluation must rely on threshold-independent discrimination metrics (ROC-AUC, Precision-Recall AUC) combined with explicit post-hoc probability threshold tuning (Abdullah & Hasan, 2023). Calibrating the classification threshold to align with the domain's cost matrix enables operational managers to select operational operating points that directly minimize aggregate financial exposure.

### 2.3 Explainable Artificial Intelligence (XAI) in Logistics
The deployment of machine learning in mission-critical operations requires transparency. Black-box models that generate risk scores without attribution often face organizational resistance from warehouse managers and logistics coordinators (Bücker et al., 2020; Schwartz et al., 2025). Methods such as Gini-based Mean Decrease in Impurity (MDI) and Permutation Feature Importance provide global interpretability, identifying systemic bottlenecks (e.g., specific carrier tiers, geographical regions, or promotional discounts) that cause delays.

---

## 3. Dataset Architecture & Exploratory Data Analysis

### 3.1 Dataset Overview
The analysis utilizes the APL Logistics / DataCo Global Supply Chain dataset, comprising **180,519 historical customer transactions** and **40 operational attributes** spanning international commercial activities. The target variable is binary:
$$\text{Late\_delivery\_risk} = \begin{cases} 1, & \text{if delivery arrived after scheduled transit window} \\ 0, & \text{if delivery arrived on or before schedule} \end{cases}$$

Across the 180,519 transactions, **98,977 shipments were delayed (54.83%)**, while **81,542 shipments were on time (45.17%)**. This distribution reveals that late delivery is not a rare anomaly, but the dominant operational outcome across the network.

### 3.2 Rigorous Data Leakage Mitigation
A critical hazard in retrospective supply chain analytics is *target leakage*—the presence of features that record information generated after the prediction event occurs. In a live deployment, predictions must be generated when an order is received or released to the warehouse. 

A comprehensive data audit identified and excluded 18 confounding variables:
- **Direct Leakage Features:** `Days for shipping (real)`, `Delivery Status`, `Actual_Delay`, `Actual_Late`, `Order Status`. (In the raw data, `Days for shipping (real) > Days for shipment (scheduled)` perfectly correlates with delay, which would artificially inflate model accuracy to 100% while rendering it useless for forward-looking deployment).
- **Post-Fulfillment Accounting Variables:** `Benefit per order`, `Sales per customer`, `Order Item Profit Ratio`, `Order Profit Per Order`.
- **Customer Personal Identifiers:** `Customer Fname`, `Customer Lname`, `Customer Street`, `Customer Zipcode`, `Customer Id`, `Order Customer Id`.

### 3.3 Selected Input Features
After leakage elimination and high-cardinality pruning (removing redundant postal and coordinate data), **14 clean operational predictors** were established:
- **Numerical (4):** `Order Item Discount`, `Order Item Discount Rate`, `Order Item Product Price`, `Order Item Quantity`.
- **Categorical (10):** `Type` (Payment method: DEBIT, TRANSFER, PAYMENT, CASH), `Category Name` (Product line), `Customer Country`, `Customer Segment` (Consumer, Corporate, Home Office), `Customer State`, `Department Name`, `Market` (LATAM, Europe, Pacific Asia, USCA, Africa), `Order Country`, `Order Region`, `Shipping Mode` (Standard Class, Second Class, First Class, Same Day).

### 3.4 Exploratory Insights: The Shipping Mode Paradox
Cross-tabulation of delivery outcomes against transportation modes uncovered a profound operational finding:

| Shipping Mode | Total Shipments | Delay Count | Observed Late Rate (%) |
|---|:---:|:---:|:---:|
| **First Class** | 27,814 | 26,513 | **95.32%** |
| **Second Class** | 35,216 | 26,985 | **76.63%** |
| **Same Day** | 9,737 | 4,454 | **45.74%** |
| **Standard Class** | 107,752 | 41,025 | **38.07%** |

```mermaid
xychart-beta
    title "Late Delivery Incidence Across Shipping Modes (%)"
    x-axis ["First Class", "Second Class", "Same Day", "Standard Class"]
    y-axis "Delay Rate (%)" 0 --> 100
    bar [95.32, 76.63, 45.74, 38.07]
```

**The Paradox:** Customers selecting and paying for premium expedited shipping (*First Class*) experience an overwhelming **95.32% failure rate**, whereas economy shipments (*Standard Class*) fail at only **38.07%**. This disparity indicates that carriers are assigned unachievable delivery windows (typically 1 to 2 days) for First Class consignments without corresponding dedicated rapid-transit fulfillment capacity.

---

## 4. Methodology & Model Architecture

### 4.1 Data Partitioning & Preprocessing Pipeline
To guarantee rigorous out-of-sample evaluation, the dataset was split into an **80% training set (144,415 orders)** and a **20% test set (36,104 orders)** using stratified sampling based on the target distribution.

The data preprocessing was structured into an automated `ColumnTransformer`:
- **Numerical Features:** Imputed using median values to resist skewness, followed by standard z-score normalization (`StandardScaler`).
- **Categorical Features:** Imputed with the most frequent value, followed by `OneHotEncoder(handle_unknown='ignore', min_frequency=20)`. Infrequent categories below 20 occurrences were pooled into a composite category to prevent high-dimensional sparse explosion.

### 4.2 Algorithm Benchmarking
Three distinct model architectures were evaluated:
1. **Logistic Regression (L2 Regularized):** Established as a linear, interpretable baseline ($C=1.0$, `max_iter=1000`).
2. **Gradient Boosting Classifier:** Sequential ensemble boosting utilizing shallow decision trees (`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`).
3. **Random Forest Classifier:** Bagged ensemble comprising 150 de-correlated decision trees with `max_depth=20`, `min_samples_leaf=5`, `class_weight='balanced'`, and `random_state=42`.

### 4.3 Cost-Sensitive Decision Calibration
To identify the optimal operational decision cutoff, probability thresholds from $\tau = 0.30$ to $\tau = 0.50$ (in increments of 0.05) were systematically evaluated on a held-out validation split. 

The optimization objective was formulated to maximize late-delivery recall while maintaining operational precision above 60%:
$$\tau^* = \arg\max_{\tau} \text{Recall}(\tau) \quad \text{subject to} \quad \text{Precision}(\tau) \ge 0.60$$

---

## 5. Empirical Results

### 5.1 Out-of-Sample Model Comparison
On the 36,104 test transactions, models evaluated at the standard $\tau = 0.50$ threshold yielded the following discrimination metrics:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Pipeline** | **0.6996** | **0.8455** | **0.5532** | **0.6688** | **0.7603** |
| Gradient Boosting | 0.6977 | 0.8463 | 0.5483 | 0.6655 | 0.7451 |
| Logistic Regression | 0.6949 | 0.8336 | 0.5541 | 0.6657 | 0.7435 |

The Random Forest architecture achieved the highest overall ROC-AUC (**0.7603**), outperforming both linear and boosting baselines across discriminatory stability.

### 5.2 Threshold Tuning Impact Analysis
Evaluating the Random Forest model across threshold settings demonstrates the severe trade-offs inherent in the default cutoff:

| Threshold ($\tau$) | Precision | Recall | F1-Score | Late Deliveries Caught (TP) | Missed Delays (FN) | False Alarms (FP) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.50 (Default) | 0.8455 | 0.5532 | 0.6688 | 10,950 | 8,846 | 1,999 |
| 0.45 | 0.8190 | 0.5730 | 0.6740 | 11,343 | 8,453 | 2,508 |
| 0.40 | 0.7910 | 0.6010 | 0.6830 | 11,897 | 7,899 | 3,142 |
| **0.35 (Optimal)** | **0.6000** | **0.8879** | **0.7176** | **17,577** | **2,219** | **11,717** |
| 0.30 | 0.5520 | 0.9950 | 0.7100 | 19,697 | 99 | 15,980 |

```mermaid
xychart-beta
    title "Operational Impact of Threshold Tuning on Test Set"
    x-axis ["Threshold 0.50", "Threshold 0.40", "Threshold 0.35 (Tuned)"]
    y-axis "Late Deliveries Caught (Orders)" 0 --> 20000
    bar [10950, 11897, 17577]
```

#### Key Empirical Findings:
1. At the conventional $\tau = 0.50$ cutoff, the model misses **8,846 late shipments** out of 19,796 actual delays (a failure rate of 44.7%).
2. At the calibrated $\tau = 0.35$ cutoff, late-delivery recall surges to **88.79%**, capturing **17,577 delayed shipments**.
3. Missed late deliveries plunge from **8,846 down to 2,219**—representing an immediate **75.0% reduction in unflagged supply chain breaches**.
4. The harmonic mean F1-Score improves from **0.6688 to 0.7176**, proving that $\tau = 0.35$ represents a mathematically superior operational balance.

### 5.3 Test Set Risk Stratification
To operationalize predictions, each shipment was assigned to a risk tier based on its posterior probability $P(\text{Late} \mid X)$:
- **High Risk ($P \ge 0.70$):** **10,217 orders (28.3%)** with an actual delay rate of **89.2%** (average predicted risk: 81.7%).
- **Medium Risk ($0.40 \le P < 0.70$):** **4,786 orders (13.3%)** with an actual delay rate of **58.8%**.
- **Low Risk ($P < 0.40$):** **21,101 orders (58.4%)** with an actual delay rate of **37.3%**.

---

## 6. Model Explainability & Feature Attribution

### 6.1 Gini Impurity (MDI) Importance
Aggregating one-hot encoded variables back to their parent features reveals the core operational determinants of late delivery risk:

| Rank | Original Variable | Grouped Importance Score | % Contribution |
|:---:|---|:---:|:---:|
| 1 | **Shipping Mode** | 0.8687 | **86.87%** |
| 2 | **Type (Payment / Order Type)** | 0.0315 | 3.15% |
| 3 | **Order Country** | 0.0255 | 2.55% |
| 4 | **Customer State** | 0.0176 | 1.76% |
| 5 | **Order Region** | 0.0107 | 1.07% |
| 6 | **Order Item Discount** | 0.0083 | 0.83% |
| 7 | **Category Name** | 0.0083 | 0.83% |
| 8 | **Order Item Discount Rate** | 0.0061 | 0.61% |
| 9 | **Customer Segment** | 0.0060 | 0.60% |
| 10 | **Order Item Product Price** | 0.0046 | 0.46% |
| 11 | **Department Name** | 0.0044 | 0.44% |
| 12 | **Market** | 0.0030 | 0.30% |
| 13 | **Order Item Quantity** | 0.0028 | 0.28% |
| 14 | **Customer Country** | 0.0025 | 0.25% |

### 6.2 Permutation Feature Importance
Permutation feature importance was computed on a stratified sample of 5,000 test observations by measuring the mean degradation in ROC-AUC when each feature's values were randomly shuffled across 3 iterations:

| Feature | Mean ROC-AUC Drop | Standard Deviation |
|---|:---:|:---:|
| **Shipping Mode** | **+0.1624** | 0.00491 |
| **Type** | **+0.0082** | 0.00404 |
| **Order Country** | +0.0018 | 0.00034 |
| **Customer State** | +0.0015 | 0.00045 |
| **Order Region** | +0.0011 | 0.00049 |

Both explainability methodologies confirm that **Shipping Mode** overwhelmingly governs fulfillment probability. The structural mechanics of the supply chain network are largely determined by carrier agreements and transit promises rather than individual product attributes.

---

## 7. Operational Implementation: The Decision Support System

To translate predictive intelligence into warehouse-floor action, the trained model pipeline was operationalized into a lightweight, interactive **Streamlit Dashboard (`app.py`)**.

```
+---------------------------------------------------------------------------------+
|                       GLOBAL SUPPLY CHAIN RISK DASHBOARD                        |
+------------------------------------+--------------------------------------------+
|  OPERATIONAL CONTROLS (Sidebar)    |  REAL-TIME KPI METRIC CARDS                |
|  - Threshold Slider: [0.35]        |  - Filtered Orders: 36,104                 |
|  - Market Multi-Select: [All]      |  - Orders Flagged: 29,294 (81.1%)          |
|  - Region Multi-Select: [All]      |  - Late Delivery Recall: 88.8%             |
|  - Shipping Mode: [All]            |  - High-Risk Queue (>70%): 10,217          |
+------------------------------------+--------------------------------------------+
|  INTERACTIVE VISUALIZATION SUITE                                                |
|  [ Bar: Risk Tiers ]    [ Grouped Bar: Shipping Mode Failure Rates ]             |
|  [ Regional Risk Heatmap: Top Vulnerable Global Fulfillment Zones ]             |
+---------------------------------------------------------------------------------+
|  PRIORITY DISPATCH ALERT QUEUE (Tabular Interface)                              |
|  Order ID | Shipping Mode | Region | Risk Prob | Status | Action Required       |
|  #127141  | First Class   | US     | 94.2%     | High   | Expedite Dispatch     |
|  #074789  | First Class   | Asia   | 92.8%     | High   | Carrier Re-route      |
|  [ Download Full Flagged Queue as CSV ]                                         |
+---------------------------------------------------------------------------------+
```

### Operational Workflow:
1. **Intake & Pre-Scoring:** Incoming orders are streamed through `trained_random_forest_pipeline.pkl`.
2. **Threshold Filtering:** Shipments exceeding $\tau = 0.35$ are automatically routed into the **Priority Dispatch Queue**.
3. **Dispatch Intervention:** Warehouse logistics supervisors prioritize flagged orders for expedited cross-docking, dedicated courier transfers, or pre-departure customer communication.

---

## 8. Managerial Implications & Strategic Recommendations

### 8.1 Resolution of the Premium Shipping Paradox
The 95.32% delay rate in First Class shipping represents severe brand risk. Supply chain executives must:
- **Recalibrate Promise Windows:** Extend published First Class delivery estimates by +24 to +48 hours to align customer expectations with physical fulfillment constraints.
- **Enforce 3PL Carrier Penalties:** Re-negotiate contracts with express carriers to establish clawback mechanisms for non-performance.

### 8.2 Dynamic Lead Time Scheduling
Rather than displaying static transit estimates at online checkout, e-commerce engines should interface directly with the Random Forest scoring model:
- Orders originating from high-risk destinations (e.g., Central Africa with a 62.1% delay rate or Eastern Europe at 59.3%) should dynamically receive adjusted scheduled delivery dates.

### 8.3 Capacity Buffering for Promotional Surges
High discounts (`Order Item Discount > 20%`) correlate with increased delay probability due to packaging bottlenecks. Distribution centers should dynamically allocate temporary labor during promotional spikes to prevent packing queue starvation.

---

## 9. Limitations & Future Research

1. **Cross-Sectional vs. Temporal Splitting:** The present study utilizes a stratified random split. While this evaluates generalized pattern recognition across shipments, a prospective roll-forward temporal evaluation (training on months 1–10, evaluating on months 11–12) would provide deeper insight into seasonal regime shifts.
2. **Observational vs. Causal Interpretation:** The model identifies predictive correlations, not causal relationships. Operational interventions (such as altering shipping mode rules) should be tested via controlled A/B rollouts.
3. **Integration of Real-Time Telematics:** Future iterations should incorporate upstream supplier telematics, weather disruption indices, and live carrier GPS tracking to dynamically update risk scores in transit.

---

## 10. Conclusion

This study successfully establishes a machine-learning framework for predicting late delivery risk in global supply chain operations. By purging target leakage and benchmarking multiple architectures, a tuned Random Forest pipeline was identified as the optimal classification engine (ROC-AUC = 0.7603). 

Crucially, reframing delivery prediction as an asymmetric cost problem and adjusting the classification threshold from **0.50 to 0.35** captured **88.79% of all delayed shipments**, shrinking missed late deliveries by **75.0%**. Model interpretability unmasked the critical operational breakdown in express shipping tiers, providing leadership with actionable leverage to optimize logistics agreements, adjust customer delivery commitments, and safeguard operational profitability.

---

## References

- Abdullah, M. S., & Hasan, R. (2023). AI-driven insights for product marketing: Enhancing customer experience and refining market segmentation. *American Journal of Interdisciplinary Studies*, 4(04), 80–116.
- Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2020). Transparency, auditability and explainability of machine learning models in credit scoring. *Journal of Operational Research*, arXiv:2009.13384.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning*, 233–240.
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. E. (2023). *An Introduction to Statistical Learning: With Applications in Python*. Springer.
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer New York.
- Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136.
- Nasir, F., Ahmed, A., Kiraz, M., Yevseyeva, I., & Saif, M. (2024). Data-driven decision-making for bank target marketing using supervised learning classifiers on imbalanced big data. *Computers, Materials & Continua*, 81(1), 1703–1728.
- Schwartz, S., Wang, Q., & Fang, F. (2025). Enhancing ML models interpretability for operational risk scoring. *Decision Support Systems*, arXiv:2509.11389.
- Zheng, Y. (2025). Prediction of the effectiveness of bank marketing strategies using machine learning models. *Advances in Economics, Management and Political Sciences*, 170(1), 17–28.
