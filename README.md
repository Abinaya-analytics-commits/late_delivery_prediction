# Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations

**Unified Mentors — Project 1 Final Submission Package**

---

## 📌 Executive Overview
Late deliveries represent a critical vulnerability in global supply chain operations, directly triggering contractual SLA penalties, elevated customer service costs, and customer churn. In this project, an end-to-end Machine Learning Early Warning System was developed on **180,519 global shipments** from APL Logistics / DataCo. 

By framing late delivery prevention as a **cost-sensitive classification problem**, a Random Forest classifier was trained on 14 non-leaking operational variables. Evaluating prediction cutoffs revealed that tuning the decision threshold to **0.35** captures **88.79% of all delayed shipments** (reducing missed late deliveries by **75%** compared to the standard 0.50 cutoff).

---

## 📊 Key Empirical Findings & Metrics
- **Dataset Scale:** 180,519 shipments | Target: `Late_delivery_risk` | Baseline Late Rate: **54.83%**
- **Leading Model:** Random Forest Pipeline (`n_estimators=150`, `max_depth=20`, `min_samples_leaf=5`, `class_weight='balanced'`)
- **Model Discrimination:** **ROC-AUC = 0.7603** (Test set)
- **Threshold Optimization (0.35 vs. 0.50):**
  - **Recall:** Surged from **55.2% to 88.79%** (Caught **17,577 delayed shipments**)
  - **Missed Delays:** Decreased from **8,864 down to 2,219**
  - **F1-Score:** Improved from **0.6685 to 0.7176**
- **The First-Class Shipping Paradox:** 
  - **First Class orders exhibit a 95.32% delay rate** (avg predicted probability: 88.4%).
  - **Second Class:** 76.63% delay rate.
  - **Standard Class:** 38.07% delay rate.
  - *Managerial Insight:* Customers paying the highest shipping premiums experience the poorest fulfillment reliability due to overly tight, unachievable promise windows.

---

## 🗂️ Project Repository Structure

```
project_1/
│
├── README.md                                  # Comprehensive project documentation
├── requirements.txt                           # Python dependencies
├── app.py                                     # Interactive Streamlit operations dashboard
├── trained_random_forest_pipeline.pkl         # Exported full scikit-learn pipeline (.pkl)
├── execute_technical_pipeline.py              # Reproducible pipeline execution script
│
├── Research_Paper_Late_Delivery_Risk.md       # Full Academic Research Paper
├── Executive_Summary_Late_Delivery_Risk.md    # 2-Page C-Suite Management Brief
├── UM_Late_Delivery_Risk_Prediction.ipynb     # Complete Jupyter Notebook
│
├── data/
│   ├── APL_Logistics.csv                      # Raw supply chain dataset (180,519 rows)
│   ├── UM_scored_test_orders_tuned.csv        # 36,104 test orders with probabilities & tiers
│   └── late_delivery_alert_queue.csv          # Prioritized dispatch intervention queue
│
├── results/
│   ├── UM_final_model_evaluation.csv          # Model comparison table (0.50 vs 0.35 threshold)
│   ├── permutation_importance_results.csv     # Permutation feature importance (ROC-AUC drop)
│   ├── RF_grouped_feature_importance.csv      # Grouped MDI importance by original feature
│   └── RF_feature_importance_full.csv         # Full encoded feature importances
│
└── figures/
    ├── feature_importance.png                 # Grouped feature importance bar chart
    ├── permutation_importance.png             # Permutation importance bar chart
    ├── shipping_mode_delay_risk.png           # Delay rate by shipping mode comparison
    ├── confusion_matrix_threshold_035.png     # Test set confusion matrix at 0.35 threshold
    └── risk_category_distribution.png         # Order counts across High, Medium, Low tiers
```

---

## 🚀 How to Run the Interactive Dashboard

1. **Activate your environment & install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Dashboard Capabilities:**
   - Calibrate the real-time threshold slider (default: `0.35`).
   - Filter by Market, Region, and Shipping Mode.
   - Inspect live KPIs (Total Orders, Recall %, Delayed Volume, High-Risk Orders).
   - Review the **Priority Dispatch Alert Queue** and download flagged shipments as a CSV for operations teams.

---

## 🎓 Deliverables Summary
1. **Academic Research Paper:** `Research_Paper_Late_Delivery_Risk.md` / `.pdf`
2. **Executive Summary:** `Executive_Summary_Late_Delivery_Risk.md` / `.pdf`
3. **Interactive Streamlit Dashboard:** `app.py`
4. **Trained Random Forest Pipeline:** `trained_random_forest_pipeline.pkl`
5. **Scored Test Datasets & Alert Queue:** `data/`
6. **Model Metrics & Charts:** `results/` & `figures/`
