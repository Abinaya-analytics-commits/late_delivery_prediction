# Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations

**Author:** Data Science & Logistics Operations Group  
**Project:** Unified Mentors — Project 1 Completion Report  
**Date:** November 2026  

---

## Abstract

# Abstract

Delivery punctuality is an essential determinant of customer satisfaction, service-level agreement (SLA) compliance, and operational efficiency in global supply chain networks. This study presents a machine learning framework for predicting late delivery risk using a benchmark dataset of 180,519 historical shipments from APL Logistics / DataCo.

To reduce data leakage, post-fulfillment variables such as actual shipping days, delivery status, and order fulfillment information were excluded. Fourteen operational attributes were selected as the primary predictive features. Logistic Regression, Gradient Boosting, and Random Forest algorithms were evaluated. Among the three models, the Random Forest classifier achieved the highest ROC-AUC score of 0.7595, with an accuracy of 0.6996 on the test dataset.

The study further examines classification threshold adjustment to improve the identification of potentially delayed shipments. At a probability threshold of 0.35, the tuned Random Forest model identified 17,577 delayed shipments, achieving a recall of 88.79%, precision of 60.00%, and F1-score of 0.7161. This lower threshold improves the model's ability to identify at-risk shipments, although it also produces more false-positive alerts.

Feature importance analysis identifies Shipping Mode as the most influential predictor of late delivery risk, highlighting the relevance of transportation-related attributes in predicting delivery outcomes.

Finally, the model results are presented through an interactive Streamlit decision-support dashboard built using pre-scored test data. The dashboard enables users to explore delivery risk, adjust classification thresholds, filter shipment records, and examine flagged orders.

**Keywords:** Supply Chain Analytics, Late Delivery Prediction, Random Forest, Classification Threshold Optimization, Model Interpretability, Logistics Decision Support Systems.

---

## 1. Introduction

### 1.1 Operational Context
Globalized commerce has transformed supply chain networks into complex, highly interdependent ecosystems characterized by multi-tiered supplier tiers, geographically dispersed distribution centers, and diverse last-mile carrier arrangements (Kuhn & Johnson, 2013). Within this paradigm, delivery timeliness has transitioned from a competitive differentiator to a mandatory operational requirement. Contractual enterprise agreements commonly enforce stringent On-Time, In-Full (OTIF) service levels, penalizing delivery breaches through direct contractual chargebacks, freight expedited re-routing expenses, and damaged business reputations (Zheng, 2025).

Despite advances in Enterprise Resource Planning (ERP) and Transportation Management Systems (TMS), supply chain operations remain vulnerable to disruptions. Order backlogs, warehouse processing congestion, regional transportation capacity imbalances, and inaccurate transit time estimations frequently cascade into fulfillment failures. In typical enterprise logistics datasets, late delivery incidence often exceeds 50%, highlighting widespread systemic mismatches between scheduled commitments and physical fulfillment capabilities.

## 1.2 Operational Trade-Off in Late-Delivery Prediction
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
The deployment of machine learning in mission-critical operations requires transparency. Black-box models that generate risk scores without attribution often face organizational resistance from warehouse managers and logistics coordinators (Bücker et al., 2020). Methods such as Gini-based Mean Decrease in Impurity (MDI) and Permutation Feature Importance provide global interpretability, identifying systemic bottlenecks (e.g., specific carrier tiers, geographical regions, or promotional discounts) that cause delays.

---

## 3. Dataset Architecture & Exploratory Data Analysis

### 3.1 Dataset Overview
The analysis utilizes the APL Logistics / DataCo Global Supply Chain dataset, comprising **180,519 historical customer transactions** and **40 operational attributes** spanning international commercial activities. The target variable is binary:
$$\text{Late\_delivery\_risk} = \begin{cases} 1, & \text{if delivery arrived after scheduled transit window} \\ 0, & \text{if delivery arrived on or before schedule} \end{cases}$$

Across the 180,519 transactions, **98,977 shipments were delayed (54.83%)**, while **81,542 shipments were on time (45.17%)**. This distribution reveals that late delivery is not a rare anomaly, but the dominant operational outcome across the network.

### 3.2 Rigorous Data Leakage Mitigation

A critical challenge in retrospective supply chain analytics is target leakage, which occurs when a predictive model uses information that would not be available at the time a prediction is made. In a practical deployment, delivery risk must be estimated using information available when an order is received or prepared for dispatch.

To minimize data leakage, the dataset was examined and variables containing post-fulfillment information were excluded from the predictive feature set.

The excluded variables included:

**Direct Leakage Features:** Days for shipping (real), Delivery Status, Actual_Delay, Actual_Late, and Order Status.

**Post-Fulfillment Accounting Variables:** Benefit per order, Sales per customer, Order Item Profit Ratio, and Order Profit Per Order.

**Customer Personal Identifiers:** Customer Fname, Customer Lname, Customer Street, Customer Zipcode, Customer Id, and Order Customer Id.

These variables were excluded because they either contained information generated after the delivery outcome, represented information unrelated to the intended prediction stage, or served as personal identifiers rather than useful operational predictors.

Removing these variables helped establish a predictive feature set based on information intended to be available before dispatch.

### 3.3 Selected Input Features

Following data leakage mitigation and feature selection, 14 operational predictors were retained for model development. These variables represent order characteristics, customer information, geographical attributes, product details, and shipping arrangements.

**Numerical Features (4):**
1. Order Item Discount
2. Order Item Discount Rate
3. Order Item Product Price
4. Order Item Quantity

**Categorical Features (10):**
1. Type
2. Category Name
3. Customer Country
4. Customer Segment
5. Customer State
6. Department Name
7. Market
8. Order Country
9. Order Region
10. Shipping Mode

The numerical variables capture quantitative order and product characteristics, while the categorical variables represent customer segments, geographical locations, product classifications, and transportation arrangements.

These 14 features were used as inputs to the machine learning preprocessing and classification pipeline.

### 3.4 Exploratory Insights: The Shipping Mode Paradox

An exploratory analysis of delivery outcomes across different shipping modes revealed substantial differences in observed late-delivery rates.

| Shipping Mode | Total Shipments | Delay Count | Observed Late Rate (%) |
|---|---:|---:|---:|
| First Class | 27,814 | 26,513 | 95.32% |
| Second Class | 35,216 | 26,985 | 76.63% |
| Same Day | 9,737 | 4,454 | 45.74% |
| Standard Class | 107,752 | 41,025 | 38.07% |

**Key Observation:**

First Class shipments recorded the highest observed late-delivery rate at 95.32%, followed by Second Class shipments at 76.63%. Same Day shipments recorded a late-delivery rate of 45.74%, while Standard Class shipments had the lowest rate at 38.07%.

This pattern presents an interesting operational paradox: shipments associated with premium shipping categories experienced higher observed delay rates than Standard Class shipments.

**Operational Interpretation:**

The differences may indicate a mismatch between scheduled delivery commitments and actual transportation performance. Other possible contributing factors include geographical constraints, carrier capacity, transportation network conditions, and differences in order handling.

However, the dataset does not establish the specific causes of these delays. Further investigation into carrier performance, scheduled transit times, and regional transportation conditions would be necessary to identify the underlying operational reasons.

These findings highlight Shipping Mode as an important variable for understanding and predicting delivery risk.
---

## 4. Methodology & Model Architecture


### 4.1 Data Partitioning & Preprocessing Pipeline

To ensure reliable out-of-sample evaluation, the dataset was divided into training and testing subsets using an 80:20 split. Stratified sampling was applied based on the target variable to maintain a similar distribution of late and non-late deliveries across both subsets. The split used a random state of 42 to ensure reproducibility.

The resulting datasets contained 144,415 training observations and 36,104 testing observations.

Data preprocessing was implemented using an automated ColumnTransformer within a machine learning pipeline.

**Numerical Features:** Missing values were handled using median imputation, followed by standardization using StandardScaler.

**Categorical Features:** Missing values were replaced using the most frequent category. OneHotEncoder was then applied with handle_unknown='ignore' and min_frequency=20 to handle previously unseen and infrequent categories.

Integrating preprocessing within the pipeline ensured that the transformations were consistently applied during model training and prediction.

### 4.2 Algorithm Benchmarking

Three supervised machine learning algorithms were selected to evaluate their ability to distinguish between late and non-late deliveries.

**Logistic Regression:** Used as a linear classification baseline, with max_iter=1000 and random_state=42.

**Gradient Boosting Classifier:** Implemented using 100 estimators, a learning rate of 0.1, maximum tree depth of 3, and random_state=42.

**Random Forest Classifier:** Implemented as an ensemble of 150 decision trees, with maximum depth of 20, minimum samples per leaf of 5, class_weight='balanced', and random_state=42.

All three algorithms were trained using the selected 14 operational features and evaluated on the held-out test dataset.

Model performance was assessed using accuracy, precision, recall, F1-score, and ROC-AUC. These metrics provided a comparative assessment of classification performance and the models' ability to distinguish between delivery outcomes.

### 4.3 Recall-Oriented Threshold Calibration 

The default classification threshold of 0.50 was examined to determine whether a lower probability cutoff could improve the identification of potentially delayed shipments.

To select an appropriate threshold, the training dataset was further divided into fitting and validation subsets using stratified sampling. A fresh copy of the Random Forest pipeline was trained on the fitting subset, and predicted probabilities were generated for the validation subset.

Thresholds of 0.30, 0.35, 0.40, 0.45, and 0.50 were evaluated.

The selection objective was to maximize late-delivery recall while maintaining a minimum precision of 0.60:

\[
\tau^* = \arg\max_{\tau} \operatorname{Recall}(\tau)
\]

Subject to:

\[
\operatorname{Precision}(\tau) \geq 0.60
\]

Based on the validation results, a threshold of 0.35 was selected. This threshold achieved approximately 89.1% recall while maintaining precision at approximately 60%.

The selected threshold was subsequently evaluated on the held-out test dataset to assess its performance on unseen observations.

---

## 5. Empirical Results

### 5.1 Out-of-Sample Model Comparison

On the 36,104 test transactions, the three machine learning models were evaluated at the standard classification threshold of \(\tau = 0.50\). Their performance was assessed using accuracy, precision, recall, F1-score, and ROC-AUC.

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Pipeline** | **0.6996** | **0.8455** | **0.5532** | **0.6688** | **0.7595** |
| Gradient Boosting | 0.6977 | 0.8463 | 0.5483 | 0.6655 | 0.7451 |
| Logistic Regression | 0.6949 | 0.8336 | 0.5541 | 0.6657 | 0.7435 |

The Random Forest classifier achieved the highest overall accuracy of 0.6996 and ROC-AUC of 0.7595, demonstrating the strongest overall discrimination between late and non-late deliveries among the evaluated models.

Gradient Boosting achieved an ROC-AUC of 0.7451, while Logistic Regression achieved 0.7435. Although Logistic Regression recorded a slightly higher recall of 0.5541 compared with Random Forest's 0.5532, Random Forest provided the best overall balance of classification performance.

Based on its ROC-AUC and overall evaluation results, Random Forest was selected for subsequent classification threshold optimization.

### 5.2 Threshold Tuning Impact Analysis

The Random Forest classifier initially used the standard classification threshold of 0.50. However, in supply chain operations, failing to identify a potentially delayed shipment may have greater operational consequences than generating a false-positive alert.

To improve the identification of potentially delayed shipments, classification thresholds ranging from 0.30 to 0.50 were evaluated on the validation dataset. The selection objective was to maximize recall while maintaining a minimum precision of 0.60.

**Validation Threshold Comparison**

| Threshold | Precision | Recall | F1-Score | True Positives | Missed Late Deliveries | False Alarms |
|---|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.552 | 0.995 | 0.710 | 15,756 | 80 | 12,772 |
| **0.35** | **0.600** | **0.891** | **0.717** | **14,114** | **1,722** | **9,427** |
| 0.40 | 0.791 | 0.601 | 0.683 | 9,513 | 6,323 | 2,511 |
| 0.45 | 0.819 | 0.573 | 0.674 | 9,079 | 6,757 | 2,007 |
| 0.50 | 0.842 | 0.549 | 0.665 | 8,696 | 7,140 | 1,628 |

The threshold of 0.30 achieved the highest recall but failed to satisfy the minimum precision requirement. The threshold of 0.35 met the specified precision constraint while providing the highest recall among the evaluated thresholds satisfying that constraint. Consequently, 0.35 was selected for final test-set evaluation.

**Final Test-Set Evaluation**

The selected threshold was subsequently evaluated on the independent test dataset containing 36,104 transactions.

| Metric | Threshold 0.50 | Threshold 0.35 |
|---|---:|---:|
| Precision | 0.8455 | 0.6000 |
| Recall | 0.5532 | 0.8879 |
| F1-Score | 0.6688 | 0.7161 |
| True Positives (TP) | 10,950 | 17,577 |
| False Negatives (FN) | 8,846 | 2,219 |
| False Positives (FP) | 1,999 | 11,717 |

**Key Empirical Findings:**

1. At the default threshold of 0.50, the model identified 10,950 delayed shipments while missing 8,846 actual delays.
2. At the selected threshold of 0.35, the model identified 17,577 delayed shipments, increasing recall from 55.32% to 88.79%.
3. Missed late deliveries decreased from 8,846 to 2,219, representing an approximately 74.9% reduction in false negatives.
4. Precision decreased from 84.55% to 60.00%, while false-positive alerts increased from 1,999 to 11,717.
5. The F1-score improved from 0.6688 to approximately 0.7161.

Overall, the threshold of 0.35 provides a more recall-oriented classification strategy for identifying potentially delayed shipments. Although this approach generates additional false-positive alerts, it substantially reduces the number of delayed shipments that go undetected.

### 5.3 Test Set Risk Stratification

To support operational interpretation of model predictions, each shipment in the test dataset was assigned to a risk category based on its predicted probability of late delivery.

Three risk categories were established using probability thresholds:

- **High Risk (P ≥ 0.70):** Shipments with a predicted late-delivery probability of 0.70 or above.
- **Medium Risk (0.40 ≤ P < 0.70):** Shipments with predicted probabilities between 0.40 and 0.70.
- **Low Risk (P < 0.40):** Shipments with predicted probabilities below 0.40.

The resulting distribution across the 36,104 test transactions is presented below.

| Risk Category | Number of Orders | Actual Late Deliveries | Actual Late Rate | Average Predicted Risk |
|---|---:|---:|---:|---:|
| High | 10,367 | 9,215 | 88.9% | 81.5% |
| Medium | 4,506 | 2,652 | 58.9% | 55.2% |
| Low | 21,231 | 7,929 | 37.3% | 35.3% |

**Key Findings:**

1. High-risk shipments accounted for approximately 28.7% of the test dataset, with an actual late-delivery rate of 88.9%.
2. Medium-risk shipments represented approximately 12.5% of test observations, with an actual late-delivery rate of 58.9%.
3. Low-risk shipments accounted for approximately 58.8% of test observations, with an actual late-delivery rate of 37.3%.

The results demonstrate a clear increase in observed late-delivery rates across the low, medium, and high predicted-risk categories. This indicates that the Random Forest model provides useful risk stratification for prioritizing shipments that may require additional operational attention.

---

# 6. Model Explainability & Feature Attribution

### 6.1 Gini Impurity (MDI) Feature Importance

Mean Decrease in Impurity (MDI) was used to examine the contribution of individual encoded features to the Random Forest model's predictive decisions.

The feature importance output revealed that Shipping Mode categories dominated the model's importance distribution.

| Encoded Feature | Importance Score | Contribution (%) |
|---|---:|---:|
| Shipping Mode – Standard Class | 0.42464 | 42.464 |
| Shipping Mode – First Class | 0.28495 | 28.495 |
| Shipping Mode – Second Class | 0.13104 | 13.104 |
| Shipping Mode – Same Day | 0.03268 | 3.268 |
| Type – TRANSFER | 0.01974 | 1.974 |
| Order Item Discount | 0.00773 | 0.773 |
| Order Item Discount Rate | 0.00588 | 0.588 |
| Type – DEBIT | 0.00541 | 0.541 |
| Order Item Product Price | 0.00448 | 0.448 |
| Type – PAYMENT | 0.00390 | 0.390 |

The four Shipping Mode categories collectively account for approximately 87.33% of the model's total encoded feature importance. This indicates that transportation mode plays a dominant role in the model's prediction process.

### 6.2 Permutation Feature Importance

Permutation Feature Importance was calculated using a representative sample of 5,000 test observations. The analysis used three repetitions with ROC-AUC as the evaluation metric.

The method measures the decrease in model performance when the values of an individual feature are randomly shuffled.

| Feature | Mean ROC-AUC Drop | Standard Deviation |
|---|---:|---:|
| Shipping Mode | 0.23533 | 0.00491 |
| Type | 0.02490 | 0.00404 |
| Order Country | 0.01110 | 0.00034 |
| Customer State | 0.00944 | 0.00045 |
| Order Region | 0.00445 | 0.00049 |
| Customer Segment | 0.00336 | 0.00128 |
| Customer Country | 0.00230 | 0.00027 |
| Market | 0.00106 | 0.00027 |
| Order Item Product Price | 0.00080 | 0.00031 |
| Order Item Discount | 0.00074 | 0.00050 |
| Order Item Quantity | 0.00038 | 0.00003 |
| Department Name | 0.00015 | 0.00028 |
| Order Item Discount Rate | -0.00051 | 0.00009 |
| Category Name | -0.00118 | 0.00048 |

**Interpretation:**

Shipping Mode recorded the highest permutation importance, with a mean ROC-AUC decrease of 0.23533. Type, Order Country, and Customer State were the next most influential features, although their contributions were considerably smaller.

The slightly negative permutation scores for Order Item Discount Rate and Category Name indicate that shuffling these features did not reduce model performance on the evaluated sample.

Both MDI and permutation importance identify Shipping Mode as the most influential predictor. These findings highlight the importance of transportation-related attributes in the model's assessment of late-delivery risk.

---

## 7. Operational Implementation: The Decision Support System

To translate predictive intelligence into operational insights, the trained Random Forest model's predictions were presented through a lightweight, interactive **Streamlit Dashboard (`app.py`)**.

The dashboard provides a decision-support interface through which users can explore pre-scored test orders, adjust prediction thresholds, identify potentially delayed shipments and examine delivery risk patterns.

### Dashboard Interface

```text
+---------------------------------------------------------------------------------+
|                  GLOBAL SUPPLY CHAIN RISK DASHBOARD                              |
+------------------------------------+--------------------------------------------+
|  OPERATIONAL CONTROLS (Sidebar)    |  DYNAMIC KPI METRIC CARDS                  |
|  - Threshold Slider: [0.35]        |  - Filtered Orders: 36,104                 |
|  - Market Multi-Select: [All]      |  - Orders Flagged: 29,294 (81.1%)          |
|  - Region Multi-Select: [All]      |  - Late Delivery Recall: 88.8%             |
|  - Shipping Mode: [All]            |  - High-Risk Orders (>70%): 10,367        |
|                                    |  - Active Threshold: 0.35                 |
+------------------------------------+--------------------------------------------+
|  INTERACTIVE VISUALIZATION SUITE                                                |
|  [ Bar: Predicted Risk Categories ]                                              |
|  [ Grouped Bar: Actual Late Rate vs. Average Predicted Risk by Shipping Mode ]  |
|  [ Bar: Top 10 Regions by Actual Late Delivery Rate ]                            |
+---------------------------------------------------------------------------------+
|  PRIORITY DISPATCH ALERT QUEUE                                                  |
|  Shipping Mode | Market | Region | Customer State | Risk Probability | Status  |
|  [ Orders sorted by predicted late-delivery probability ]                       |
|  [ Download Full High-Risk Alert Queue as CSV ]                                 |
+---------------------------------------------------------------------------------+
```

*Note: KPI values represent the default full-dataset view at the 0.35 threshold. The dashboard recalculates the displayed metrics when filters or the threshold are changed.*

### Operational Workflow

1. **Prediction and Data Preparation:** The trained Random Forest model generates late-delivery probabilities for the test dataset. These predictions are stored in a scored CSV file, which is loaded by the Streamlit application.

2. **Threshold-Based Alert Identification:** The dashboard compares each order's predicted late-delivery probability against the selected threshold (\(\tau = 0.35\)). Orders meeting or exceeding the threshold are included in the priority alert queue.

3. **Risk Monitoring and Filtering:** Users can examine flagged orders by market, region and shipping mode, while visualisations provide insights into risk categories, shipping-mode delivery rates and regional delay patterns.

4. **Priority Review and Export:** The alert queue displays orders in descending order of predicted risk. Users can download the complete flagged queue as a CSV file for further analysis and operational review.

### Practical Operational Application

The dashboard demonstrates how machine learning predictions can support supply chain decision-making. In a practical warehouse or logistics environment, teams could use the flagged-order list to investigate potentially delayed shipments and consider interventions such as expedited dispatch, carrier re-routing or proactive customer communication.

The current implementation is an analytical prototype based on pre-scored test data. It demonstrates the potential for operational decision support but does not automatically process live incoming orders or execute dispatch interventions.

## 8. Managerial Implications & Strategic Recommendations

### 8.1 Resolution of the Premium Shipping Paradox

The 95.32% late-delivery rate observed for First Class shipping in the cleaned dataset indicates a significant operational performance concern. Although premium shipping is generally associated with faster delivery expectations, the observed delay rate suggests a gap between service expectations and actual fulfilment outcomes.

Supply chain executives should consider the following measures:

- **Recalibrate Promise Windows:** Review existing First Class delivery estimates against actual fulfilment performance. Any proposed extension, such as 24–48 hours, should first be validated through historical delivery data and controlled operational testing.
- **Strengthen Third-Party Logistics Monitoring:** Introduce regular carrier performance reviews using delivery reliability, delay frequency and service-level agreement compliance. Contractual performance incentives or penalties may be considered where supported by documented carrier performance.
- **Prioritise High-Risk Shipments:** Use predicted delivery probabilities to identify First Class shipments requiring closer monitoring and early operational review.

### 8.2 Dynamic Lead Time Scheduling

The regional analysis identifies differences in late-delivery patterns across geographical locations. Central Africa recorded an actual late-delivery rate of 62.1%, while Eastern Europe recorded 59.3% in the test dataset.

Rather than relying exclusively on static delivery estimates, e-commerce platforms could explore integrating predictive risk information into delivery planning.

Recommended measures include:

- **Risk-Aware Delivery Estimates:** Consider regional delivery-risk patterns when displaying estimated delivery dates during checkout.
- **Regional Performance Monitoring:** Regularly review delivery outcomes across regions to identify locations with consistently elevated delay rates.
- **Predictive Scheduling Integration:** Integrate the trained Random Forest model into future order-management systems to support dynamic delivery estimates based on order characteristics and predicted risk.

These recommendations should be validated using additional operational data and repeated delivery-performance assessments before being implemented at scale.

### 8.3 Capacity Buffering for Promotional Surges

Promotional campaigns and periods of increased order volume can create additional demands on fulfilment operations. Although the present analysis does not establish that high discounts directly cause packaging bottlenecks or delivery delays, capacity planning remains an important managerial consideration.

Distribution centres could consider the following measures:

- **Monitor Order Volume:** Track incoming order volumes and delivery-risk patterns during promotional campaigns.
- **Plan Workforce Capacity:** Evaluate temporary staffing requirements based on anticipated order volumes and actual warehouse workload.
- **Monitor Fulfilment Bottlenecks:** Collect additional operational data on picking time, packing queues, warehouse capacity and dispatch delays to identify potential causes of late deliveries.
- **Use Predictive Alerts:** Incorporate high-risk order alerts into operational monitoring so that potentially delayed shipments can be reviewed before dispatch.

These measures can help organisations develop a more proactive approach to fulfilment planning and delivery-risk management.

---

## 9. Limitations & Future Research

1. **Cross-Sectional vs. Temporal Splitting:** The present study utilizes a stratified random split. While this evaluates generalized pattern recognition across shipments, a prospective roll-forward temporal evaluation (training on months 1–10, evaluating on months 11–12) would provide deeper insight into seasonal regime shifts.
2. **Observational vs. Causal Interpretation:** The model identifies predictive correlations, not causal relationships. Operational interventions (such as altering shipping mode rules) should be tested via controlled A/B rollouts.
3. **Integration of Real-Time Telematics:** Future iterations should incorporate upstream supplier telematics, weather disruption indices, and live carrier GPS tracking to dynamically update risk scores in transit.

---

## 10. Conclusion

This study establishes a machine-learning framework for predicting late delivery risk in global supply chain operations. By addressing potential target leakage, applying appropriate preprocessing techniques and benchmarking multiple classification algorithms, the Random Forest model emerged as the best-performing model among those evaluated, achieving an accuracy of 69.96% and a ROC-AUC of 0.7595.

A key contribution of this study is the use of threshold optimisation to address the importance of identifying potentially delayed shipments. By lowering the classification threshold from **0.50 to 0.35**, the model achieved a test-set recall of **88.79%**, identifying a substantially larger proportion of delayed shipments. Compared with the default threshold, missed late deliveries decreased by approximately **74.9%**, although this improvement involved an increase in false-positive alerts.

Model interpretability further highlighted Shipping Mode as the most influential predictor of delivery risk, identifying an important area for operational investigation. The findings also demonstrate the value of regional delivery analysis and risk-based monitoring in supply chain operations.

Finally, the interactive Streamlit dashboard translates model predictions into an accessible decision-support interface, enabling users to examine high-risk orders, adjust classification thresholds and export flagged records. Although the current implementation uses pre-scored test data rather than live incoming orders, it provides a foundation for future integration into operational supply chain systems.

Overall, this study demonstrates how machine learning can support proactive delivery-risk identification and informed logistics decision-making. Future work involving temporal validation, real-time operational data and deployment integration could further strengthen the framework's practical applicability.

---

## References

Abdullah, M. S., & Hasan, R. (2023). AI-driven insights for product marketing: Enhancing customer experience and refining market segmentation. *American Journal of Interdisciplinary Studies*, 4(04), 80–116.

Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2020). *Transparency, auditability and explainability of machine learning models in credit scoring*. arXiv preprint arXiv:2009.13384.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136.

Nasir, F., Ahmed, A., Kiraz, M., Yevseyeva, I., & Saif, M. (2024). Data-driven decision-making for bank target marketing using supervised learning classifiers on imbalanced big data. *Computers, Materials & Continua*, 81(1), 1703–1728.

Zheng, Y. (2025). Prediction of the effectiveness of bank marketing strategies using the XGBoost model. *Advances in Economics, Management and Political Sciences*.
