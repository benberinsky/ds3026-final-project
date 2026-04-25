# DS 3026: Final Project

### Reproducible Notebook

**Group:** Ben Berinsky, Will Wert, Carissa Chen, Mason Nicoletti

<br>

---

## I. Problem Formulation

- Clearly describe your dataset and context
- Define one inferential question (e.g., effect, difference, parameter)
- Define one predictive task (e.g., classification or regression)
- Clearly explain the difference between the two

<br>

### Dataset and Context

**Dataset:** Students_Performance_dataset.csv
- Contains student demographic information
- Many features
- Target Variable: GPA

**Context:**

Student performance is often measured through GPA, but this is impacted by a wide range of factors. Study habits, home/family life, access to resources, responsibilities outside of school, and a multitude of other influences impact grade point average. In this report, we analyze the relationship between various factors and GPA. The dataset contains information about roughly 1200 computer science and engineering students at a private institution in Bangladesh. Data was collected through an online survey form. The dataset contains 31 features, both academic and non-academic. Key fields include cumulative GPA, scholarship status, study hours per day, social media time, attendance, and more.

**Link to Data Source:**

[Students' Academic Performance Evaluation Dataset](https://data.mendeley.com/datasets/dc3797vf3t/1)

<br>

### Inference and Prediction

**Inferential Question:** 

Is there a credible difference between the mean cumulative grade point average of scholarship students and non-scholarship students for computer science and engineering students?

**Predictive Task:** 

Can we predict the cumulative grade point average of a new student based on the information present in the dataset, specifically lifestyle, demographic, and past academic information?

**Differentiation:**

The key distinction is that our inferential question focuses on estimating a specific population parameter (the true mean difference in GPA), whereas our predictive task focuses on minimizing error for individual observations (unseen students).

<br>

---

## Data Loading and Preprocessing


```python
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from scripts.data_prep import data_prep

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```


```python
# Load raw data

data_raw = pd.read_csv("./data/Students_Performance_dataset.csv")
```


```python
# Preview data

data_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>University Admission year</th>
      <th>Gender</th>
      <th>Age</th>
      <th>H.S.C passing year</th>
      <th>Program</th>
      <th>Current Semester</th>
      <th>Do you have meritorious scholarship ?</th>
      <th>Do you use University transportation?</th>
      <th>How many hour do you study daily?</th>
      <th>How many times do you seat for study in a day?</th>
      <th>...</th>
      <th>What is you interested area?</th>
      <th>What is your relationship status?</th>
      <th>Are you engaged with any co-curriculum activities?</th>
      <th>With whom you are living with?</th>
      <th>Do you have any health issues?</th>
      <th>What was your previous SGPA?</th>
      <th>Do you have any physical disabilities?</th>
      <th>What is your current CGPA?</th>
      <th>How many Credit did you have completed?</th>
      <th>What is your monthly family income?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018</td>
      <td>Male</td>
      <td>24</td>
      <td>2016</td>
      <td>BCSE</td>
      <td>12</td>
      <td>Yes</td>
      <td>No</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>Data Schince</td>
      <td>Single</td>
      <td>Yes</td>
      <td>Bachelor</td>
      <td>No</td>
      <td>2.68</td>
      <td>No</td>
      <td>3.15</td>
      <td>75</td>
      <td>25000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>Male</td>
      <td>22</td>
      <td>2020</td>
      <td>BCSE</td>
      <td>4</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>Event management</td>
      <td>Single</td>
      <td>Yes</td>
      <td>Family</td>
      <td>No</td>
      <td>2.68</td>
      <td>No</td>
      <td>3.15</td>
      <td>36</td>
      <td>100000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020</td>
      <td>Female</td>
      <td>21</td>
      <td>2019</td>
      <td>BCSE</td>
      <td>5</td>
      <td>No</td>
      <td>No</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>Software</td>
      <td>Single</td>
      <td>No</td>
      <td>Bachelor</td>
      <td>No</td>
      <td>2.68</td>
      <td>No</td>
      <td>3.15</td>
      <td>50</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021</td>
      <td>Male</td>
      <td>20</td>
      <td>2020</td>
      <td>BCSE</td>
      <td>4</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>Artificial Intelligence</td>
      <td>Single</td>
      <td>No</td>
      <td>Bachelor</td>
      <td>Yes</td>
      <td>2.68</td>
      <td>No</td>
      <td>3.15</td>
      <td>36</td>
      <td>62488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>Male</td>
      <td>22</td>
      <td>2019</td>
      <td>BCSE</td>
      <td>4</td>
      <td>Yes</td>
      <td>No</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>Software</td>
      <td>Relationship</td>
      <td>No</td>
      <td>Bachelor</td>
      <td>Yes</td>
      <td>2.68</td>
      <td>No</td>
      <td>3.15</td>
      <td>36</td>
      <td>50000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
# Apply data prep function

data = data_prep(data_raw)
```


```python
# Preview cleaned data

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>study_hours</th>
      <th>study_seatings</th>
      <th>social_media_hours</th>
      <th>attendance</th>
      <th>skill_hours</th>
      <th>semester_gpa</th>
      <th>cumulative_gpa</th>
      <th>credits</th>
      <th>family_income</th>
      <th>...</th>
      <th>skills_programming</th>
      <th>skills_software_development</th>
      <th>skills_web_development</th>
      <th>interest_area_data_science</th>
      <th>interest_area_hardware</th>
      <th>interest_area_machine_learning</th>
      <th>interest_area_networking</th>
      <th>interest_area_other</th>
      <th>interest_area_software</th>
      <th>interest_area_ui/ux</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>90.0</td>
      <td>2</td>
      <td>2.68</td>
      <td>3.15</td>
      <td>75</td>
      <td>25000</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>96.0</td>
      <td>2</td>
      <td>2.68</td>
      <td>3.15</td>
      <td>36</td>
      <td>100000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>80.0</td>
      <td>1</td>
      <td>2.68</td>
      <td>3.15</td>
      <td>50</td>
      <td>50000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>88.0</td>
      <td>1</td>
      <td>2.68</td>
      <td>3.15</td>
      <td>36</td>
      <td>62488</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>80.0</td>
      <td>1</td>
      <td>2.68</td>
      <td>3.15</td>
      <td>36</td>
      <td>50000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
# View features

data.columns
```




    Index(['age', 'study_hours', 'study_seatings', 'social_media_hours',
           'attendance', 'skill_hours', 'semester_gpa', 'cumulative_gpa',
           'credits', 'family_income', 'english_proficiency_basic',
           'english_proficiency_intermediate', 'health_issues_no',
           'health_issues_yes', 'health_issues_no', 'scholarship_yes',
           'gender_male', 'transportation_yes', 'disabilities_yes',
           'skills_cyber_security', 'skills_machine_learning', 'skills_networking',
           'skills_other', 'skills_programming', 'skills_software_development',
           'skills_web_development', 'interest_area_data_science',
           'interest_area_hardware', 'interest_area_machine_learning',
           'interest_area_networking', 'interest_area_other',
           'interest_area_software', 'interest_area_ui/ux'],
          dtype='object')



---

## II. Likelihood and Estimation

- Specify a statistical model (e.g., Bernoulli, Normal, Logistic)
- Write down the likelihood function (conceptually or mathematically)
- Compute or estimate parameters using MLE
- Visualize or interpret how the likelihood behaves

**Inferential Question:** Is there a credible difference between the mean cumulative grade point average of scholarship students and non-scholarship students for computer science and engineering students?


```python
# Separate scholarship and non-scholarship students
non_scholarship = data[data['scholarship_yes'] == 0]["cumulative_gpa"]
scholarship = data[data['scholarship_yes'] == 1]["cumulative_gpa"]
```


```python
# Visualize gpa distributions

plt.figure()
plt.hist(non_scholarship, bins=20, alpha=0.6, density=True, label="Non-scholarship")
plt.hist(scholarship, bins=20, alpha=0.6, density=True, label="Scholarship")
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.legend()
plt.savefig("./figures/gpa_distribution.png")
plt.show()
```


    
![png](final_project_files/final_project_12_0.png)
    


### Specify a statistical model

Statistical Model: **Normal Distribution**

The distribution of GPA for students is best modeled using a Normal distribution. After separating students into two groups, one for scholarship students and the other for non-scholarship students, we can assume both groups follow a Normal distribution with their own mean and standard deviation.

### Likelihood Function

**Conceptually:** What is the plausibility of the parameter values for the mean GPA and standard deviation for non-scholarship students and the mean GPA and standard deviation for scholarship students given the observed student performance data?

Let *x*<sub>*i*</sub> denote a student.

$$
x_i \sim 
\begin{cases}
\mathcal{N}(\mu_0, \sigma_0^2), & \text{if } i \in \text{group 0 (non-scholarship)} \\
\mathcal{N}(\mu_1, \sigma_1^2), & \text{if } i \in \text{group 1 (scholarship)}
\end{cases}
$$

**Mathematically:**

Likelihood Function:

$$
L(\mu_0, \mu_1, \sigma_0^2, \sigma_1^2) = 
\prod_{i \in \text{group 0}} f(x_i \mid \mu_0, \sigma_0^2)
\times
\prod_{i \in \text{group 1}} f(x_i \mid \mu_1, \sigma_1^2)
$$

Applying the Normal Density statistical model:

$$
f(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} 
\exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

### Estimate parameters using MLE


```python
# Estimate parameters using MLE

# Compute parameters
mu_0, sigma_0 = non_scholarship.mean(), non_scholarship.std(ddof=0)
mu_1, sigma_1 = scholarship.mean(), scholarship.std(ddof=0)

p_0 = len(non_scholarship) / len(data)
p_1 = len(scholarship) / len(data)

# Create a mle df
mle_df = pd.DataFrame({
    "Non-scholarship": [mu_0, sigma_0, p_0],
    "Scholarship": [mu_1, sigma_1, p_1]},
    index=['Mean', 'Standard Deviation', 'Percentage']
    )

print("Parameter Estimates:")
print(mle_df)
```

    Parameter Estimates:
                        Non-scholarship  Scholarship
    Mean                       3.134672     3.205430
    Standard Deviation         0.599609     0.905613
    Percentage                 0.561609     0.438391


### Visualize and interpret likelihood


```python
# Define a log-likelihood function

def log_likelihood(mu, data, sigma):
    log_likelihood = -0.5 * np.sum(((data - mu) / sigma)**2) - len(data) * np.log(sigma)
    return log_likelihood
```


```python
# Create a range of mu values
mu_0_vals = np.linspace(mu_0 - 1, mu_0 + 1, 100)
mu_1_vals = np.linspace(mu_1 - 1, mu_1 + 1, 100)

# Create a list of log-likelihoods
log_likelihood_mu_0 = [log_likelihood(mu, non_scholarship, sigma_0) for mu in mu_0_vals]
log_likelihood_mu_1 = [log_likelihood(mu, scholarship, sigma_0) for mu in mu_0_vals]
```


```python
# Plot likelihood values

plt.figure()

plt.plot(mu_0_vals, log_likelihood_mu_0, color='Blue', label='Non-Scholarship')
plt.axvline(mu_0, linestyle='--', color='Blue')

plt.plot(mu_1_vals, log_likelihood_mu_1, color='Red', label='Scholarship')
plt.axvline(mu_1, linestyle='--', color='Red')

plt.title("Log-Likelihood for μ Values")
plt.xlabel("GPA (μ)")
plt.ylabel("Log-Likelihood")
plt.legend()
plt.tight_layout()
plt.savefig("./figures/log_likelihood_plot.png")
plt.show()
```


    
![png](final_project_files/final_project_20_0.png)
    


#### Interpretation of Likelihood

In the visualization above, the log-likelihoods are plotted for the non-scholarship and scholarship groups. The curves trace the likelihood of true mean GPA for a student occurring at that GPA value, which runs along the x-axis. The likelihoods are maximized at the sample means (μ₀ and μ₁). Both curves are similar in shape, however, the non-scholarship curve has a sharper peak that also has a greater log-likelihood. This indicates the mean GPA for non-scholarship students was estimated with greater certainty.

---

## III. Frequentist Inference

- Construct at least one confidence interval (preferably bootstrap)
- Interpret the interval in context
- (Optional) Perform a hypothesis-style comparison if appropriate

**Important:**

- Focus on interpretation, not formulas
- Explain what uncertainty means in your problem

**Question:** Is there a statistically significant difference between the mean GPA of scholarship students and non-scholarship students in the wider student population?


```python
# Split into the two groups of interest
scholarship = data[data['scholarship_yes'] == 1]['cumulative_gpa']
non_scholarship = data[data['scholarship_yes'] == 0]['cumulative_gpa']

print(f"Scholarship students:     n = {len(scholarship)}, mean GPA = {scholarship.mean():.4f}, std = {scholarship.std():.4f}")
print(f"Non-scholarship students: n = {len(non_scholarship)}, mean GPA = {non_scholarship.mean():.4f}, std = {non_scholarship.std():.4f}")
print(f"\nObserved difference in means (scholarship - non-scholarship): {scholarship.mean() - non_scholarship.mean():.4f}")
```

    Scholarship students:     n = 523, mean GPA = 3.2054, std = 0.9065
    Non-scholarship students: n = 670, mean GPA = 3.1347, std = 0.6001
    
    Observed difference in means (scholarship - non-scholarship): 0.0708


### Normal Approximation Confidence Interval

With group sizes of over 500 students each, the Central Limit Theorem ensures the sampling distribution of the mean difference is approximately normal. This lets us use a simple **z-interval** to construct a 95% CI for the true difference in mean GPA.


```python
# --- Normal approximation confidence interval ---

n_s  = len(scholarship)
n_ns = len(non_scholarship)

mean_diff = scholarship.mean() - non_scholarship.mean()
se_diff   = np.sqrt(scholarship.var(ddof=1)/n_s + non_scholarship.var(ddof=1)/n_ns)

z_crit  = 1.96  # 95% CI, two-tailed
ci_low  = mean_diff - z_crit * se_diff
ci_high = mean_diff + z_crit * se_diff

print(f"Observed mean GPA difference (scholarship - non-scholarship): {mean_diff:.4f}")
print(f"Standard error of the difference:                            {se_diff:.4f}")
print(f"\n95% Normal Approximation CI: ({ci_low:.4f}, {ci_high:.4f})")
```

    Observed mean GPA difference (scholarship - non-scholarship): 0.0708
    Standard error of the difference:                            0.0459
    
    95% Normal Approximation CI: (-0.0192, 0.1608)


### Visualizing the Normal Approximation CI


```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: GPA distributions for both groups
ax = axes[0]
ax.hist(non_scholarship, bins=25, alpha=0.55, density=True, label='Non-Scholarship', color='steelblue')
ax.hist(scholarship, bins=25, alpha=0.55, density=True, label='Scholarship', color='darkorange')
ax.axvline(non_scholarship.mean(), color='steelblue', linestyle='--', linewidth=1.5, label=f'Non-Sch. mean = {non_scholarship.mean():.2f}')
ax.axvline(scholarship.mean(), color='darkorange', linestyle='--', linewidth=1.5, label=f'Sch. mean = {scholarship.mean():.2f}')
ax.set_title('GPA Distributions by Scholarship Status')
ax.set_xlabel('Cumulative GPA')
ax.set_ylabel('Density')
ax.legend(fontsize=8)

# Right: CI plot for the difference in means
ax2 = axes[1]
ax2.errorbar(
    x=[mean_diff], y=[0],
    xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
    fmt='o', color='black', capsize=8, capthick=2, markersize=8, label='Parametric 95% CI'
)
ax2.axvline(0, color='red', linestyle='--', linewidth=1.2, label='No difference (0)')
ax2.set_title('95% CI for Difference in Mean GPA\n(Scholarship − Non-Scholarship)')
ax2.set_xlabel('Difference in Mean GPA')
ax2.set_yticks([])
ax2.legend(fontsize=9)
ax2.set_xlim(ci_low - 0.05, ci_high + 0.05)

plt.tight_layout()
plt.savefig("./figures/mean_gpa_ci_and_distribution.png")
plt.show()
```


    
![png](final_project_files/final_project_28_0.png)
    


#### Interpretation of Normal Approximation CI

The 95% CI gives a plausible range for the **true population difference** in mean GPA between scholarship and non-scholarship students. If the interval does not include 0, the observed difference is unlikely to be explained by chance alone. If it includes 0, we cannot rule out that the true population difference is zero.

The width of the interval reflects **uncertainty due to sampling** — with a larger or more representative sample we would expect a narrower interval.

### Bootstrap Confidence Interval

The normal approximation CI relies on the CLT holding for our sample sizes. A **bootstrap CI** makes no distributional assumptions at all — it estimates the sampling distribution directly by resampling from the data with replacement.

If the two intervals agree closely, it gives us additional confidence in the result.


```python
# --- Bootstrap confidence interval ---

np.random.seed(42)
n_bootstrap = 5000
bootstrap_diffs = np.empty(n_bootstrap)

for i in range(n_bootstrap):
    # Resample with replacement from each group
    boot_s  = np.random.choice(scholarship, size=n_s, replace=True)
    boot_ns = np.random.choice(non_scholarship, size=n_ns, replace=True)
    bootstrap_diffs[i] = boot_s.mean() - boot_ns.mean()

boot_ci_low, boot_ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

print(f"Bootstrap mean of differences:     {bootstrap_diffs.mean():.4f}")
print(f"95% Bootstrap CI: ({boot_ci_low:.4f}, {boot_ci_high:.4f})")
print()
print(f"Parametric CI:    ({ci_low:.4f}, {ci_high:.4f})")
print(f"Bootstrap CI:     ({boot_ci_low:.4f}, {boot_ci_high:.4f})")
```

    Bootstrap mean of differences:     0.0708
    95% Bootstrap CI: (-0.0197, 0.1575)
    
    Parametric CI:    (-0.0192, 0.1608)
    Bootstrap CI:     (-0.0197, 0.1575)



```python
# Visualize the bootstrap sampling distribution
plt.figure(figsize=(9, 5))
plt.hist(bootstrap_diffs, bins=60, color='slateblue', alpha=0.7, edgecolor='white', label='Bootstrap differences')
plt.axvline(boot_ci_low,  color='green', linestyle='--', linewidth=2, label=f'Bootstrap CI lower ({boot_ci_low:.3f})')
plt.axvline(boot_ci_high, color='green', linestyle='--', linewidth=2, label=f'Bootstrap CI upper ({boot_ci_high:.3f})')
plt.axvline(mean_diff,    color='black', linestyle='-',  linewidth=2, label=f'Observed difference ({mean_diff:.3f})')
plt.axvline(0,            color='red',   linestyle=':',  linewidth=1.5, label='No difference (0)')
plt.title('Bootstrap Sampling Distribution of Mean GPA Difference\n(Scholarship − Non-Scholarship)')
plt.xlabel('Difference in Mean GPA')
plt.ylabel('Frequency')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("./figures/bootstrap_distribution.png")
plt.show()
```


    
![png](final_project_files/final_project_32_0.png)
    


#### Interpretation of Bootstrap CI

The histogram shows the **bootstrap sampling distribution** — a simulated picture of how much the estimated mean GPA difference could vary across different random samples. The green dashed lines mark the 2.5th and 97.5th percentiles, forming the 95% bootstrap CI. The close agreement between the parametric CI and the bootstrap CI strengthens our conclusion: neither approach is sensitive to assumptions the other might be violating.

### What Uncertainty Means Here

Frequentist inference quantifies **uncertainty due to random sampling**. Our dataset is a snapshot of students from one institution; we want to draw conclusions about all students in the broader population. The confidence intervals capture that uncertainty.

The observed difference is small in absolute terms (~0.07 GPA points). Both CIs span a range that includes zero, meaning the data are consistent with there being no real difference in the population. Statistical significance and practical significance are not the same thing — the frequentist framework helps us resist over-interpreting a small observed difference in a finite sample as a definitive truth about the population.


```python
# Final summary printout
print("=" * 60)
print("FREQUENTIST INFERENCE SUMMARY")
print("=" * 60)
print(f"Observed difference (scholarship - non-scholarship):  {mean_diff:+.4f} GPA points")
print()
print(f"95% Normal Approximation CI:  ({ci_low:.4f}, {ci_high:.4f})")
print(f"95% Bootstrap CI:             ({boot_ci_low:.4f}, {boot_ci_high:.4f})")
print()
if ci_low < 0 < ci_high:
    print("Conclusion: Both CIs contain 0 — we cannot rule out that the true")
    print("difference in population mean GPA is zero.")
else:
    print("Conclusion: Both CIs exclude 0 — statistically significant difference.")
```

    ============================================================
    FREQUENTIST INFERENCE SUMMARY
    ============================================================
    Observed difference (scholarship - non-scholarship):  +0.0708 GPA points
    
    95% Normal Approximation CI:  (-0.0192, 0.1608)
    95% Bootstrap CI:             (-0.0197, 0.1575)
    
    Conclusion: Both CIs contain 0 — we cannot rule out that the true
    difference in population mean GPA is zero.


---

## IV. Bayesian Estimation

- Choose a reasonable prior (explain your choice)
- Compute or simulate a posterior (grid or simple approach is fine)
- Report a credible interval
- Compare your Bayesian results to frequentist results


### Inferential

**Inferential Question:** Is there a credible difference between the mean cumulative grade point average of scholarship students and non-scholarship students for computer science and engineering students?

### Computation Process


```python
# getting scholarship yes count
scholarship_count = data['scholarship_yes'].sum()
no_scholarship_count = len(data) - scholarship_count 
print(f"Number of students with scholarship: {scholarship_count}, without scholarship: {no_scholarship_count}")
```

    Number of students with scholarship: 523, without scholarship: 670


Based on the normal-normal conjugate, 

Posterior Distribution of Parameter: 
$$\theta | x \sim N\left( \frac{\frac{n\bar{x}}{\sigma^2} + \frac{\mu}{\tau^2}}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}, \frac{1}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}} \right)$$

Bayesian Point Estimate: 
$$\hat{\theta}_{BE} = \left( \frac{n\tau^2}{n\tau^2 + \sigma^2} \right)\bar{x} + \left( \frac{\sigma^2}{n\tau^2 + \sigma^2} \right)\mu$$

Where: 
- $\bar{x}$: Sample Mean (The observed difference in our dataset)
- $n$: Sample Size.$\sigma^2$: Data Variance (Assumed known from the sample)
- $\mu$: Prior Mean (Our initial belief, e.g., $0$ for no difference)
- $\tau^2$: Prior Variance (How certain we are of our prior belief)
- $\frac{1}{\sigma^2}$: Sample Precision (inverse of variance)
- $\frac{1}{\tau^2}$: Prior Precision (inverse of variance)

Finding parameter (diff between scholarship and no scholarship GPA)


```python
#scholarship_students = student_data[student_data['scholarship'] == 'Yes']
scholarship_students = data[data['scholarship_yes'] == 1]
no_scholarship_students = data[data['scholarship_yes'] == 0]
mean_scholarship_gpa = scholarship_students['cumulative_gpa'].mean()
mean_no_scholarship_gpa = no_scholarship_students['cumulative_gpa'].mean()
gpa_diff = mean_scholarship_gpa - mean_no_scholarship_gpa
print(f"Mean GPA for scholarship students: {mean_scholarship_gpa:.2f}")
print(f"Mean GPA for non-scholarship students: {mean_no_scholarship_gpa:.2f}")
print(f"Difference in mean GPA: {gpa_diff:.2f}")
```

    Mean GPA for scholarship students: 3.21
    Mean GPA for non-scholarship students: 3.13
    Difference in mean GPA: 0.07


Finding values for variables


```python
x_bar = gpa_diff
n = len(data)
# using sample variance as an estimate for population variance
sigma_sq = data['cumulative_gpa'].var()
```


```python
sigma_sq
```




    np.float64(0.5631604483930309)



For our prior distribution we assume there is no difference between cumulative GPA for students on scholarship and not on scholarship. This means that our prior belief is there is no effect of scholarship status on cumulative GPA. Because of this we use $\mu$ = 0 and $\sigma$ = 0.5.  


```python
# setting prior beliefs
mu = 0 
tau_sq = 0.5**2 

# calculating precision (1 / Variance)
prior_precision = 1 / tau_sq
sample_precision = n / sigma_sq
posterior_precision = prior_precision + sample_precision
```

Computing posterior metrics 
- post_mean = [(sample_mean * sample_prec) + (prior_mean * prior_prec)] / total_prec


```python
# mean
post_mu = ((x_bar * sample_precision) + (mu * prior_precision) ) / posterior_precision

# variance/std
post_var = 1 / posterior_precision
post_std = np.sqrt(post_var)
```

Now that posterior metrics have been calculated we can compute the credible interval


```python
post_std
```




    np.float64(0.021706320784384486)




```python
lower, upper = norm.interval(0.95, loc=post_mu, scale=post_std)

print(f"Posterior Mean: {post_mu:.4f}")
print(f"95% Credible Interval: ({lower:.4f}, {upper:.4f})")
```

    Posterior Mean: 0.0706
    95% Credible Interval: (0.0281, 0.1132)



```python
x = np.linspace(post_mu - 4*post_std, post_mu + 4*post_std, 1000)
y = norm.pdf(x, loc=post_mu, scale=post_std)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Posterior Distribution', color='#2c3e50', linewidth=2)
plt.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=1.5, label='No Difference (0)')
plt.fill_between(x, y, where=(x >= lower) & (x <= upper), color='#3498db', alpha=0.25, label='95% Credible Interval')
plt.title('Posterior Distribution of GPA Difference (Bayesian Inference)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Difference in Mean GPA (Scholarship - No Scholarship)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.legend(framealpha=0.9, fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("./figures/bayesian_posterior_distribution.png")
plt.show()
```


    
![png](final_project_files/final_project_54_0.png)
    


### Interpretation

The posterior mean we found was 0.07 and the 95% credible interval was [0.028, 0.113]. Since the 95% credible interval does not contain 0 we conclude that there is an effect of scholarship status on cumulative GPA for college students. This means that there is a 95% posterior probability that the true difference of cumulative GPA for college students who are on scholarship vs. those who are not on scholarship (scholarship - non scholarship) is between 0.028 and 0.113. These findings show that there is strong posterior evidence that students on scholarship have higher GPAs.

### Predictive

To what extent can a Bayesian linear regression model accurately forecast individual student GPAs using scholarship status, study hours, and demographic factors as predictors?

### Building Model


```python
student_data = data.dropna()
X = student_data.drop(['cumulative_gpa', 'semester_gpa'], axis=1)
y = student_data['cumulative_gpa']
```


```python
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fitting bayesian ridge model
model = BayesianRidge()
model.fit(X_train, y_train)

# Generating predictions and uncertainty estimates
y_pred, y_std = model.predict(X_test, return_std=True)

# Evaluating
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")
```

    Root Mean Squared Error: 0.6066
    R-squared Score: 0.1051


### Prediction Interval


```python
# Select a random sample student from test set 
sample_index = np.random.choice(len(y_pred))
individual_pred = y_pred[sample_index]
individual_std = y_std.iloc[sample_index]

# Calculate the 95% Prediction Interval
lower_pi = individual_pred - (1.96 * individual_std)
upper_pi = individual_pred + (1.96 * individual_std)

print(f"Prediction for Student {sample_index}:")
print(f"Expected Cumulative GPA: {individual_pred:.2f}")
print(f"95% Prediction Interval: ({lower_pi:.2f}, {upper_pi:.2f})")
print(f"Actual GPA: {y_test.iloc[sample_index]:.2f}")
```

    Prediction for Student 35:
    Expected Cumulative GPA: 3.03
    95% Prediction Interval: (1.42, 4.64)
    Actual GPA: 2.01


Generally, our prediction intervals are fairly wide and not very informative. This is because the model performed poorly.

---

# V. Resampling and Simulation

- Implement a bootstrap (for mean, proportion, or model)
- Explain what the variability in your results means

**Research Question:** Is there a credible difference between the mean GPA of scholarship students and non-scholarship students in the wider student population

A single boostrap is implemented to perform a hypothesis test.

H0: There is no difference between mean GPA of scholarship and nonscholarship students

Ha: There is a difference between mean GPA of scholarship and nonscholarship students

After bootstrapping to simulate different groups of students from the original sample, the mean GPA between scholar and nonscholarships students are compared for each set, then form a 95% confidence interval to determine if there is a significance.


```python
scholar = data[data['scholarship_yes']==1]['cumulative_gpa'].reset_index(drop=True)
nonscholar = data[data['scholarship_yes']==0]['cumulative_gpa'].reset_index(drop=True)

n_scholar=len(scholar)
n_nonscholar=len(nonscholar)
iterations=1000
rng=np.random.default_rng(2478)
means=np.full(iterations,np.nan)
for i in range(iterations):
  idx_s=rng.integers(0,n_scholar,size=n_scholar)
  idx_ns=rng.integers(0,n_nonscholar,size=n_nonscholar)

  mean_s=scholar.iloc[idx_s].mean()
  mean_ns=nonscholar.iloc[idx_ns].mean()
  means[i]=mean_s-mean_ns
ci=np.percentile(means,[2.5,97.5])
print(f"Confidence Interval: {ci}")
```

    Confidence Interval: [-0.01637045  0.16358325]



```python
print(f"Average GPA Difference: {means.mean()}")
plt.figure(figsize=(10,8))
plt.hist(means,bins=30)
plt.axvline(ci[0], color='green', label='95% CI Lower')
plt.axvline(ci[1], color='green', label='95% CI Upper')
plt.title('Bootstrap Distribution of Mean GPA Difference')
plt.xlabel('GPA Difference (Scholarship - Non-Scholarship)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("./figures/resampling_bootstrap_distribution.png")
plt.show()
```

    Average GPA Difference: 0.07210575711309605



    
![png](final_project_files/final_project_67_1.png)
    


### Interpretation

With confidence intervals, if 0 is contained in the interval, it means there is no statistically significant difference. The interval of -0.0232 and 0.157 contains 0, meaning we fail to reject H0 and there is no statistical significance between GPAs of scholarship and nonscholarship students.

The variability in the results represents the overall uncertainty in the model. Here, the dataset only provides a snapshot of the overall national or global student population.

---

## VI. Prediction and Model Evaluation

- Fit at least one predictive model (e.g., logistic regression)
- Use train/test split or cross-validation
- Evaluate performance:
  - AUC / ROC (classification)
  - MSE (regression)
- Explain results in plain language

**Prediction Question:** Is it possible to predict GPA for a new student?


```python
# Separate features and target

X = data.drop(columns=['semester_gpa', 'cumulative_gpa'])
y = data['cumulative_gpa']
```


```python
# Perform train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3026)
```

### Linear Regression


```python
# Standardize features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
# Fit linear regression model

lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)
```




<style>#sk-container-id-6 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-6 {
  color: var(--sklearn-color-text);
}

#sk-container-id-6 pre {
  padding: 0;
}

#sk-container-id-6 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-6 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-6 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-6 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-6 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-6 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-6 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-6 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-6 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-6 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-6 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-6 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-6 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-6 div.sk-label label.sk-toggleable__label,
#sk-container-id-6 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-6 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-6 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-6 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-6 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-6 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-6 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-6 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-6 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-6 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" checked><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LinearRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>




```python
# Evaluate model performance on test set

y_pred = lr_model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"R2 Score: {round(r2, 4)}")
print(f"MAE: {round(mae, 4)}")
print(f"MSE: {round(mse, 4)}")
print(f"RMSE: {round(rmse, 4)}")
```

    R2 Score: 0.0711
    MAE: 0.5115
    MSE: 0.5726
    RMSE: 0.7567



```python
# Evaluate feature importance

feature_importance_lr = pd.DataFrame({
    "feature": X.columns,
    "coefficient": abs(lr_model.coef_)
})

feature_importance_lr = feature_importance_lr.sort_values(by='coefficient', ascending=False)

print(feature_importance_lr)
```

                                 feature  coefficient
    6                            credits     0.246186
    10                  health_issues_no     0.171776
    20                      skills_other     0.163416
    13                   scholarship_yes     0.137054
    11                 health_issues_yes     0.132205
    29            interest_area_software     0.094564
    4                         attendance     0.072522
    0                                age     0.057930
    19                 skills_networking     0.050325
    3                 social_media_hours     0.041514
    16                  disabilities_yes     0.041110
    24        interest_area_data_science     0.037951
    21                skills_programming     0.033866
    30               interest_area_ui/ux     0.033613
    17             skills_cyber_security     0.032259
    14                       gender_male     0.030310
    5                        skill_hours     0.028531
    27          interest_area_networking     0.026000
    22       skills_software_development     0.023840
    15                transportation_yes     0.023430
    18           skills_machine_learning     0.022008
    28               interest_area_other     0.017616
    25            interest_area_hardware     0.016960
    7                      family_income     0.016559
    12                  health_issues_no     0.016280
    9   english_proficiency_intermediate     0.015801
    8          english_proficiency_basic     0.012893
    23            skills_web_development     0.009970
    26    interest_area_machine_learning     0.008540
    1                        study_hours     0.007957
    2                     study_seatings     0.004750



```python
# Visualize feature importance 

top_features_lr = feature_importance_lr.head(8)

plt.figure()
plt.barh(top_features_lr['feature'], top_features_lr['coefficient'], color="green")
plt.title("Linear Regression Feature Importance")
plt.xlabel("Coefficient")
plt.gca().invert_yaxis()
plt.savefig("./figures/regression_feature_importance.png")
plt.show()
```


    
![png](final_project_files/final_project_78_0.png)
    


#### Interpretation

The Linear Regression model performs poorly at predicting the GPA of unseen students. The model achieves a low R2 score of 0.07, meaning only 7% of the variance in GPA can be explained by features included in the model. This indicates that the model is not effective at predicting GPA reliably. The model scored an RMSE of 0.76, meaning that typical prediction had an error of 0.76 points. This is high considering GPA runs on a scale of 0 - 4.0.

The most influential features in the linear regression model are:
1) number of credits taken
2) presence of health issues
3) student is pursuing skills in the "other" category
4) scholarship status of student
This ties into our inferential question, where it is statistically significant that a student's scholarship status is associated with their GPA.

### Random Forest


```python
# Create random forest object

rf_model = RandomForestRegressor(random_state=3026)
```


```python
# Define parameter grid for grid search

param_grid = {
    "n_estimators": [100, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
```


```python
# Implement Grid Search

grid_rf = GridSearchCV(rf_model, param_grid, n_jobs=-1, cv=4, verbose=0)

grid_rf.fit(X_train, y_train)
```




<style>#sk-container-id-7 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-7 {
  color: var(--sklearn-color-text);
}

#sk-container-id-7 pre {
  padding: 0;
}

#sk-container-id-7 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-7 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-7 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-7 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-7 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-7 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-7 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-7 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-7 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-7 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-7 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-7 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-7 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-7 div.sk-label label.sk-toggleable__label,
#sk-container-id-7 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-7 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-7 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-7 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-7 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-7 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-7 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-7 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-7 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-7 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4, estimator=RandomForestRegressor(random_state=3026),
             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [None, 10, 20],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],
                         &#x27;min_samples_split&#x27;: [2, 5, 10],
                         &#x27;n_estimators&#x27;: [100, 500]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=4, estimator=RandomForestRegressor(random_state=3026),
             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [None, 10, 20],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],
                         &#x27;min_samples_split&#x27;: [2, 5, 10],
                         &#x27;n_estimators&#x27;: [100, 500]})</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: RandomForestRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10,
                      random_state=3026)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10,
                      random_state=3026)</pre></div> </div></div></div></div></div></div></div></div></div>




```python
# Determine best model

best_rf_model = grid_rf.best_estimator_
best_score = grid_rf.best_score_
best_params = grid_rf.best_params_

print(f"R2 Score of the best model: {best_score}")
print(f"Best Parameters: {best_params}")
```

    R2 Score of the best model: 0.5016282117376969
    Best Parameters: {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}



```python
# Evaluate performance of the best model

y_pred = best_rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"R2 Score: {round(r2, 4)}")
print(f"MAE: {round(mae, 4)}")
print(f"MSE: {round(mse, 4)}")
print(f"RMSE: {round(rmse, 4)}")
```

    R2 Score: 0.6367
    MAE: 0.3627
    MSE: 0.224
    RMSE: 0.4733



```python
# Evaluate feature importance

feature_importance_rf = pd.DataFrame({
    "feature": X.columns,
    "importance": abs(best_rf_model.feature_importances_)
})

feature_importance_rf = feature_importance_rf.sort_values(by='importance', ascending=False)

print(feature_importance_rf)
```

                                 feature  importance
    6                            credits    0.695113
    4                         attendance    0.053984
    7                      family_income    0.044053
    3                 social_media_hours    0.032158
    13                   scholarship_yes    0.028707
    0                                age    0.025201
    1                        study_hours    0.017631
    5                        skill_hours    0.017340
    21                skills_programming    0.014900
    8          english_proficiency_basic    0.009333
    15                transportation_yes    0.008930
    2                     study_seatings    0.008739
    16                  disabilities_yes    0.007250
    9   english_proficiency_intermediate    0.005616
    14                       gender_male    0.005073
    22       skills_software_development    0.004424
    29            interest_area_software    0.003719
    25            interest_area_hardware    0.003327
    11                 health_issues_yes    0.002429
    10                  health_issues_no    0.002113
    23            skills_web_development    0.002091
    20                      skills_other    0.001886
    19                 skills_networking    0.001777
    17             skills_cyber_security    0.001216
    24        interest_area_data_science    0.001091
    26    interest_area_machine_learning    0.000884
    30               interest_area_ui/ux    0.000577
    18           skills_machine_learning    0.000226
    27          interest_area_networking    0.000212
    12                  health_issues_no    0.000000
    28               interest_area_other    0.000000



```python
# Visualize feature importance 

top_features_rf = feature_importance_rf.head(5)

plt.figure()
plt.barh(top_features_rf['feature'], top_features_rf['importance'], color="darkOrange")
plt.title("Random Forest Feature Importance")
plt.xlabel("Coefficient")
plt.gca().invert_yaxis()
plt.savefig("./figures/random_forest_feature_importance.png")
plt.show()
```


    
![png](final_project_files/final_project_87_0.png)
    


#### Interpretation

The random forest model scored substantially better than the linear regression model. On the test set, the best performing random forest achieved an R2 score of 0.64, indicating that it explains the majority of the variability in GPA. Additionally, this model scored a lower MSE than the regression model. The RMSE was 0.47, indicating that the typical prediction error improved to only 0.47 GPA points. The random forest performed better than linear regression on a relative scale, but it is still not a strong predictive model overall.

Looking at feature importance, the number of credits taken by the student again contributed the most to GPA prediction, although it was far more important in this model. Again, scholarship status ranked among the top five features in terms of importance, suggesting it plays a meaningful role in prediction. 

---

## VII. Regularization and Shrinkage

- Compare at least two models (e.g., simple vs regularized)
- Explain how shrinkage affects performance

**Predictive Question:** Can we improve GPA prediction over a baseline linear regression model by applying regularization?


```python
# Set up features and target

X = data.drop(columns=['semester_gpa', 'cumulative_gpa'])
y = data['cumulative_gpa']

# Train/test split — same random state as prediction notebook for consistency
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features — required for Ridge and Lasso so regularization penalizes fairly
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Training samples: {X_train_sc.shape[0]}")
print(f"Test samples:     {X_test_sc.shape[0]}")
print(f"Features:         {X_train_sc.shape[1]}")
```

    Training samples: 954
    Test samples:     239
    Features:         31


### Baseline Model: OLS Linear Regression

Ordinary Least Squares (OLS) is the unregularized baseline. It minimizes the sum of squared residuals with no penalty on coefficient size, which can lead to **overfitting** when there are many features relative to the signal in the data.


```python
ols = LinearRegression()
ols.fit(X_train_sc, y_train)
y_pred_ols = ols.predict(X_test_sc)

mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols  = r2_score(y_test, y_pred_ols)

print(f"OLS  —  MSE: {mse_ols:.4f}  |  R²: {r2_ols:.4f}")
```

    OLS  —  MSE: 0.3617  |  R²: 0.1202


### Ridge Regression (L2 Regularization)

Ridge adds an L2 penalty to the loss function — the sum of squared coefficients multiplied by a tuning parameter α:

$$\text{Loss} = \sum_{i}(y_i - \hat{y}_i)^2 + \alpha \sum_{j} \beta_j^2$$

This shrinks all coefficients toward zero but never sets them exactly to zero. We use cross-validation (`RidgeCV`) to select the best α from a grid of candidates.


```python
alphas = np.logspace(-3, 3, 100)

ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train_sc, y_train)
y_pred_ridge = ridge.predict(X_test_sc)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge  = r2_score(y_test, y_pred_ridge)

print(f"Ridge —  MSE: {mse_ridge:.4f}  |  R²: {r2_ridge:.4f}  |  Best α: {ridge.alpha_:.4f}")
```

    Ridge —  MSE: 0.3596  |  R²: 0.1253  |  Best α: 141.7474


### Lasso Regression (L1 Regularization)

Lasso adds an L1 penalty — the sum of absolute coefficient values:

$$\text{Loss} = \sum_{i}(y_i - \hat{y}_i)^2 + \alpha \sum_{j} |\beta_j|$$

Unlike Ridge, Lasso can shrink coefficients **all the way to zero**, effectively performing automatic feature selection. Features with zero coefficients are excluded from the model entirely.


```python
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso.fit(X_train_sc, y_train)
y_pred_lasso = lasso.predict(X_test_sc)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso  = r2_score(y_test, y_pred_lasso)

n_zero = np.sum(lasso.coef_ == 0)
print(f"Lasso —  MSE: {mse_lasso:.4f}  |  R²: {r2_lasso:.4f}  |  Best α: {lasso.alpha_:.4f}")
print(f"         Coefficients zeroed out: {n_zero} of {len(lasso.coef_)}")
```

    Lasso —  MSE: 0.3585  |  R²: 0.1279  |  Best α: 0.0107
             Coefficients zeroed out: 12 of 31


### Comparing Model Performance


```python
results = pd.DataFrame({
    'Model': ['OLS (baseline)', 'Ridge (L2)', 'Lasso (L1)'],
    'MSE':   [mse_ols, mse_ridge, mse_lasso],
    'R²':    [r2_ols,  r2_ridge,  r2_lasso]
})
print(results.to_string(index=False))
```

             Model      MSE       R²
    OLS (baseline) 0.361678 0.120218
        Ridge (L2) 0.359602 0.125269
        Lasso (L1) 0.358540 0.127853



```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

models = ['OLS', 'Ridge', 'Lasso']
colors = ['steelblue', 'darkorange', 'seagreen']

# MSE bar chart
axes[0].bar(models, [mse_ols, mse_ridge, mse_lasso], color=colors, alpha=0.8)
axes[0].set_title('Test MSE by Model')
axes[0].set_ylabel('Mean Squared Error')
axes[0].set_ylim(0, max(mse_ols, mse_ridge, mse_lasso) * 1.2)

# R² bar chart
axes[1].bar(models, [r2_ols, r2_ridge, r2_lasso], color=colors, alpha=0.8)
axes[1].set_title('Test R² by Model')
axes[1].set_ylabel('R²')

plt.tight_layout()
plt.savefig("./figures/regularization_model_comparison.png")
plt.show()
```


    
![png](final_project_files/final_project_100_0.png)
    


### Effect of Shrinkage on Coefficients

The clearest way to see what regularization does is to plot the coefficients from all three models side by side. Ridge shrinks coefficients toward zero but keeps all of them; Lasso pushes some all the way to zero.


```python
feature_names = X.columns.tolist()
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'OLS':     ols.coef_,
    'Ridge':   ridge.coef_,
    'Lasso':   lasso.coef_
}).set_index('Feature')

# Sort by absolute OLS coefficient for readability
coef_df = coef_df.sort_values(by='OLS', key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(8, 9))
y_pos = np.arange(len(coef_df))
width = 0.28

ax.barh(y_pos - width, coef_df['OLS'],   width, label='OLS',   color='steelblue', alpha=0.85)
ax.barh(y_pos,         coef_df['Ridge'], width, label='Ridge', color='darkorange', alpha=0.85)
ax.barh(y_pos + width, coef_df['Lasso'], width, label='Lasso', color='seagreen',  alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(coef_df.index, fontsize=8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coefficients by Model (Standardized Features)')
ax.set_xlabel('Coefficient Value')
ax.legend()
plt.tight_layout()
plt.savefig("./figures/regularization_coefficient_comparison.png")
plt.show()
```


    
![png](final_project_files/final_project_102_0.png)
    


#### Interpretation of Shrinkage

The coefficient plot shows the core effect of regularization. OLS assigns large coefficients wherever it can reduce training error, even if those large values reflect noise rather than a true signal. Ridge pulls all coefficients toward zero proportionally — large coefficients shrink more than small ones — reducing the model's sensitivity to any single feature. Lasso goes further, zeroing out the weakest predictors entirely and producing a sparser, more interpretable model.

When regularized models perform better than OLS on test data it means OLS was overfitting — learning patterns in the training set that did not generalize. When all three models perform similarly it suggests the data does not have a strong overfitting problem, and the regularization penalty is doing little harm or help.

### Bias–Variance Tradeoff

Regularization is a direct application of the **bias–variance tradeoff**. OLS is a low-bias, high-variance estimator: it fits the training data closely but can vary a lot across different samples. Adding a penalty (α) introduces bias — the model no longer fits the training data as tightly — but reduces variance, making predictions more stable on new data.

The optimal α found by cross-validation is the point where this tradeoff is best balanced: enough shrinkage to reduce overfitting, but not so much that the model becomes too simple to capture the real signal in the data.

---

## VIII. Inference vs Prediction Reflection

- Explain where inference and prediction gave different insights
- Discuss which one is more useful for your problem
- Explain how uncertainty affected your conclusion

<br>

### Reflection

Inference and prediction provided different insights. Inference focused on the overall average, examining the overall sample size to determine the difference between averages and how significant it is. The inference does tell us that there’s a statistically measurable association with GPA, even if it’s very small. Prediction focuses on the outcome of an individual student. Random Forest tries to predict a specific student’s GPA, but the deciding factor was never the scholarship status. The Linear Regression model further supports that scholarship status isn’t an influential factor in determining GPA. Inference proves a relationship exists, but Prediction proved that this relationship is weak compared to other potential factors.

Inference provides better applicable results for the overall university and student support, with the conclusion that scholarship status isn’t a primary driver of high grades. Because the difference was consistently small with low predictive power, it highlights that a student’s success depends more than just on the label, drawing attention towards other potentially hidden and correlating factors like attendance, family situation, or even distance from campus. This redirects the investigation and discussions towards them. Universities and educational institutes are generally more interested in the overall trend and relationships to help support fixes and prevent problems affecting student success, which makes inference more useful than actual prediction.

However, uncertainty was still a visible effect. In Frequentist and Resampling tests, uncertainty was wide enough that it couldn’t rule out indifference. Through including prior knowledge in the Bayesian approach, the uncertainty was reduced to shift the interval away from 0. In the predictive models, uncertainty is displayed in high RMSE and low R2, where even if the models are good, they are not able to explain and predict student performance adequately.

<br>

---
