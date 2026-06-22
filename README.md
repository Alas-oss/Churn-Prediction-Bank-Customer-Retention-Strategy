# Churn Prediction — Bank Customer Retention Strategy

## Notebook Walkthrough & Findings (Sections 1–9)

### Data Work
Before modeling, the data was audited for missing values and anomalies. Categorical features were explored and encoded, and the data was normalized and scaled to improve model efficiency. Class imbalances were also analyzed to contextualize the distribution of outcomes in subsequent visualizations.

### Exploratory Data Analysis (EDA)
Churn rates were analyzed both overall and across customer segments to identify patterns. The feature importance graph below shows that **age** is the single strongest predictor of churn, with an importance score of 0.29 — high enough to establish a meaningful threshold for identifying at-risk customers.

<img width="893" height="788" alt="Unknown" src="https://github.com/user-attachments/assets/d53b869a-0d14-4151-b3f3-fa0cf3a329ad" />

The graphs below (read left to right, top to bottom) summarize the key findings:

- **(Top Left)** Non-churners cluster around the dataset mean age of ~38. Churners skew older, predominantly falling in the **35–55** age range.
- **(Top Middle)** Lower balances are associated with lower churn risk. Balances exceeding **$100,000** correlate with a meaningfully higher likelihood of churn.
- **(Top Right)** Tenure has a relatively weak effect on churn, though customers with **6–8 years** of tenure show a slightly lower churn rate.
- **(Bottom Left)** Non-churners tend to be younger overall. The large number of outliers in that group reflects the dataset's size and the fact that older non-churning customers fall outside the typical age range for that segment. Churners, as noted, are concentrated in the middle-aged range.
- **(Bottom Middle)** Credit score has a modest effect: churned customers show a slightly lower credit score range on average.
- **(Bottom Right)** Estimated salary has a similarly limited effect, though churned customers trend marginally higher in salary.

<img width="1984" height="584" alt="Unknown" src="https://github.com/user-attachments/assets/198bd5eb-83d6-4ab6-a211-b895c313ac0f" />

<img width="1984" height="584" alt="Unknown" src="https://github.com/user-attachments/assets/d5db4e67-bf3e-4f7f-9b36-c63c741d32a8" />

### Feature Engineering

The following features were engineered to improve the model's ability to identify at-risk customers:

- **Tenure Group** — Segments customers by years with the bank
- **Activity Rate** — Flags customers with zero tenure as still active, to avoid misclassification
- **Customer Value** — Product of balance × tenure; reflects the customer's current monetary value to the bank
- **Multi-Product Customer** — Binary flag indicating whether a customer holds more than one product
- **Recent Activity** — Time elapsed since the customer's last interaction, bucketed by interval
- **Service Interaction Frequency** — How often the customer engages with bank services

### Model Training

The dataset was split 80/20 for training and testing. Three models were trained: **Logistic Regression**, **Random Forest Classifier**, and **CatBoost**. After comparing performance across relevant metrics, **CatBoost** was selected as the best-performing model.

### Evaluation

At the standard threshold of **0.50**, recall was a poor **49%** — meaning roughly half of actual churners were missed. Lowering the threshold to **0.30** raised recall to **67%**. This matters because recall directly determines how many at-risk customers can be identified and acted on before they leave.

<img width="1583" height="584" alt="Unknown-3" src="https://github.com/user-attachments/assets/427d528c-e6bf-40dc-be88-aa7df685d0f4" />

### Model Tuning

GridSearchCV (Grid Search Cross-Validation) was used to tune the model, evaluating **3,750 parameter combinations** to find the optimal configuration. The tuned model was then tested and its feature importances mapped. The top three churn drivers are:

- **Number of Products (33%)** — Every customer with 4 products churned, which heavily elevated this feature's importance
- **Age (25%)**
- **Balance (12%)**

Note: Although age ranked highest in the initial EDA, the 100% churn rate among 4-product customers caused `num_products` to dominate in the tuned model.

<img width="989" height="590" alt="Unknown-2" src="https://github.com/user-attachments/assets/beba37d4-59a4-4476-9213-e836a6b5e1f4" />

### Model Explainability

The graph below uses SHAP values to show which features push customers toward churning (right) and away from it (left).

<img width="950" height="473" alt="Unknown" src="https://github.com/user-attachments/assets/78eac7f1-81b5-407d-bd8d-882b5feffc0a" />

The following graph shows the overall importance of each feature in determining churn likelihood.

<img width="958" height="473" alt="Unknown-2" src="https://github.com/user-attachments/assets/b2704fba-74c2-4090-9cf1-6ed349b519a6" />

In the next plot, **red** indicates that a data point's value is close to the high end of the given feature's range, while **blue** indicates the low end. The x-axis represents the SHAP value — points to the right of 0 contribute toward churn, and points to the left contribute against it. For example, for `gender_Male`: blue points (Female customers) appear on the right side, indicating they are more likely to churn, while red points (Male) appear on the left, indicating lower churn risk.

<img width="824" height="499" alt="Unknown" src="https://github.com/user-attachments/assets/19cc8f13-251b-4801-b706-c8d549a3f532" />

The following waterfall plot breaks down an individual customer's churn risk. For example: even though `num_products` carries high global importance, a 49-year-old customer in the high-risk age range may have age as the dominant driver for their specific score — with product count still factored in, but weighted accordingly. This is how the model adds or subtracts from a customer's churn likelihood score on a case-by-case basis.

<img width="945" height="604" alt="Unknown" src="https://github.com/user-attachments/assets/ad779bc6-c1fd-4afa-a8b1-a36a71eaefce" />

Partial Dependence Plots (PDPs) confirmed the earlier findings: churn likelihood spikes in the **35–55** age range and falls off for older customers. Additionally, once a customer's balance crosses **$100,000**, their churn probability carries a persistent ~3% increase.

### Retention Strategy Simulation

Customers were grouped into **deciles** by predicted churn probability (Decile 1 = highest risk, Decile 10 = lowest risk). Three outreach strategies were compared:

- Contact the **top 10%** of predicted high-risk customers
- Contact the **top 25%** of predicted high-risk customers
- Contact a **random 25%** of customers

The top 25% model-driven strategy proved most effective, achieving a **2.8x Lift Multiplier Score** over the random 25% campaign of the same cost and scale.

<img width="1584" height="584" alt="Unknown-2" src="https://github.com/user-attachments/assets/3190acff-1997-49df-8e2c-b03baff49828" />

**Campaign Performance ROI Matrix**

| Group Index | Strategy             | Customers Contacted | Campaign Cost |
| :---:       | :---:                | :---:               | :---:         |
| 0           | Top 10% most likely  | 200                 | 3000.0        |
| 1           | Top 25% most likely  | 500                 | 7000.0        |
| 2           | Random 25%           | 500                 | 7000.0        |

**Loss Prevented**

| Group Index | Revenue Saved | Net ROI ($)   |
| :---:       | :---:         | :---:         |
| 0           | 5.600706e+06  | 5.597706e+06  |
| 1           | 9.518191e+06  | 9.510691e+06  |
| 2           | 3.642599e+06  | 3.635099e+06  |

The heatmap below maps churn risk and customer value by age group and number of products. Darker red indicates higher risk and higher value — these are the customers the bank stands to lose the most from. Lighter beige represents the opposite end of the spectrum.

<img width="915" height="584" alt="Unknown-3" src="https://github.com/user-attachments/assets/f7d0ad3a-7009-45b1-b48e-d82bc26c2c05" />

### Recommendations to the Bank

**Which customers should the bank prioritize?**
- Single-product customers aged **35–55**
- Inactive customers with balances exceeding **$100,000**
- Accounts based in **Germany**

**What is the expected uplift from model-driven vs. random outreach?**
- Targeting the top 25% highest-risk customers focuses effort on a group with a combined average churn rate exceeding **57%**
- Random 25% outreach yields a significantly lower prevention rate
- Model-driven targeting produces a **2.8x Lift Multiplier Score** vs. a random campaign of equivalent reach and cost

**How can the bank implement an automated churn alert system?**
- **Data Processing** — Run a Python script at the end of each preset time period
- **Scoring Deciles** — Assign a churn probability score to every account
- **Filter & Action** — Apply the **0.30 threshold** to maximize the catch rate for at-risk customers
- **CRM Integration** — Automatically move flagged accounts into the bank's CRM for follow-up
