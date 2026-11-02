# SpendSense — AI-Based Customer Spending Prediction

SpendSense is a beginner-friendly Streamlit web application that classifies customers into **Low**, **Medium** or **High** spending categories using demographic and purchasing-related information.

## Project features

- Customer CSV upload and data validation.
- Data cleaning: duplicate removal, numeric conversion, and missing-value handling through the ML pipeline.
- Exploratory analysis with six visualizations.
- Feature engineering: monthly purchase value and visits per purchase.
- Three classification baselines: Logistic Regression, Decision Tree and Random Forest.
- Weighted accuracy, precision, recall and F1-score comparison.
- Confusion matrix and model feature-importance view.
- New-customer prediction with class probabilities.
- Downloadable cleaned data, prediction output, model metrics and business insight report.

## Tools

Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn. **No Jupyter Notebook is required.** The dependency list is compatible with current Python 3.14 installations; use the provided `requirements.txt` rather than installing older package versions manually.

## Quick start (Windows PowerShell / VS Code terminal)

1. Extract the ZIP. Open the extracted `ai-customer-spending-prediction` folder in VS Code.
2. Open **Terminal → New Terminal**. Make sure the prompt is inside this project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this once in the same terminal and activate again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

6. Streamlit will print a local URL, normally `http://localhost:8501`. Open it in Chrome. To stop the app, return to the terminal and press `Ctrl + C`.

## Dataset format for your own CSV

The app accepts a CSV with these columns (column matching ignores case and spaces):

| Column | Meaning |
|---|---|
| `age` | Customer age in years |
| `gender` | Gender category |
| `annual_income` | Annual income in INR |
| `purchase_frequency` | Purchases per month |
| `avg_order_value` | Average order value in INR |
| `website_visits` | Website visits per month |
| `loyalty_years` | Customer relationship duration in years |
| `discount_usage` | Discount usage rate from 0 to 1 |
| `total_purchases` | Total purchases to date |
| `spending_category` | Low, Medium or High (recommended target column) |
| `annual_spend` | Optional annual spend in INR; used for EDA, not as a model input |

If `spending_category` is not provided, include `annual_spend`; the app will create Low/Medium/High labels using that file's lower and upper tertiles. `annual_spend` is deliberately excluded from model inputs to prevent target leakage.

The bundled `data/customer_spending_sample.csv` contains 1,500 **synthetic demonstration records**, not real customers. Use the Overview page's download button to get a fresh copy of the CSV template.

## GitHub upload

Create an empty public repository named `ai-customer-spending-prediction` under the GitHub account `PranathiM15`. Do not select README, license or `.gitignore` on the GitHub creation page because those files already exist in this project folder.

Then run these commands from the project folder in the VS Code terminal:

```powershell
git --version
git init
git branch -M main
git add .
git commit -m "Add AI customer spending prediction app"
git remote add origin https://github.com/PranathiM15/ai-customer-spending-prediction.git
git push -u origin main
```

If `git --version` says the command is not recognised, install **Git for Windows**, close and reopen VS Code, and retry. If Git says `remote origin already exists`, run this instead of `git remote add origin`:

```powershell
git remote set-url origin https://github.com/PranathiM15/ai-customer-spending-prediction.git
git push -u origin main
```

If Git asks you to sign in, complete the GitHub sign-in prompt in your browser.

## Important notes

- The app compares models on a stratified 75/25 train/test split with a fixed random seed for repeatability.
- Weighted precision, recall and F1-score are used to account for differences in class counts.
- Model results are educational estimates. Evaluate on representative, approved real data before making business decisions.
- Feature importance is an indicator of model reliance, not proof of cause and effect.
