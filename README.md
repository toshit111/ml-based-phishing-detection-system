# Phishing Website Detection

Classifying web pages as phishing or legitimate with five ML models and a Streamlit app.

- GitHub repository: `https://github.com/toshit111/ml-based-phishing-detection-system`
- Live app: `https://ml-based-phishing-detection-system-uoz8cnf5uyk3r7n9tp8nmf.streamlit.app/`

## a. Problem statement

Phishing sites imitate real websites to steal credentials and money. Checking every URL by hand is not practical, so a classifier that flags a page from measurable properties of its URL, its HTML content and its domain can act as an automatic first filter. The goal of this project is to predict whether a web page is **phishing** or **legitimate** (binary classification). Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) are trained on the same train/test split and compared on six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC. The best models are deployed in an interactive Streamlit app that evaluates the saved models on an uploaded test CSV.

## b. Dataset description

| Item | Value |
|---|---|
| Source | Kaggle, *Web page Phishing Detection Dataset* (`shashwatwork/web-page-phishing-detection-dataset`, file `dataset_phishing.csv`) |
| Problem type | Binary classification |
| Raw instances | 11,430 (5,715 phishing, 5,715 legitimate) |
| Instances used | 11,256 (174 duplicate rows removed after dropping the `url` text column) |
| Features | 87 numeric features (minimum required: 12) |
| Target | `status`: `phishing` or `legitimate` |
| Missing values | None |

Feature groups:

- **URL features:** `length_url`, `nb_dots`, `nb_hyphens`, `nb_at`, `nb_qm`, `nb_slash`, `ratio_digits_url`, `ip`, `shortening_service`, `suspecious_tld`, ...
- **Hostname and word features:** `length_hostname`, `nb_subdomains`, `longest_word_host`, `avg_word_path`, `char_repeat`, ...
- **Page content features:** `nb_hyperlinks`, `ratio_extHyperlinks`, `login_form`, `iframe`, `popup_window`, `right_clic`, `empty_title`, ...
- **Domain and external features:** `domain_age`, `domain_registration_length`, `web_traffic`, `dns_record`, `google_index`, `page_rank`.

Preprocessing and split:

- The raw `url` column is dropped (it is text, not a numeric feature) and duplicate rows are removed.
- The data is split 80/20 with stratification (`random_state=42`): 9,004 training rows and 2,252 test rows (1,143 legitimate, 1,109 phishing).
- Logistic Regression, KNN and SVM use a `StandardScaler` inside a pipeline, so scaling is saved inside each model file. Decision Tree and Naive Bayes use the raw values.
- The 2,252 held-out rows are saved as `test_data.csv`. This is the file to upload in the app.
- `phishing` is treated as the positive class for Precision, Recall, F1 and AUC.

## c. GitHub Repository Link

`<PASTE YOUR GITHUB REPO LINK HERE>`

Files in the repository:

- `app.py`: the Streamlit app. It loads the saved models and never retrains.
- `train.py`: trains the five models, saves them and writes `test_data.csv`.
- `metrics.py`: metric code shared by `train.py` and `app.py`.
- `requirements.txt`: dependencies for deployment.
- `README.md`: this file.
- `test_data.csv`: the 2,252 held-out test rows.
- `data/dataset_phishing.csv`: the full dataset.
- `model/`: the five saved `.joblib` models.
- `results/metrics.csv`: the metrics table produced by `train.py`.

To run it locally:

```
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## d. Models used

Results on the 2,252 test rows (Precision, Recall and F1 are for the `phishing` class):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9405 | 0.9813 | 0.9396 | 0.9396 | 0.9396 | 0.8810 |
| Decision Tree | 0.9352 | 0.9274 | 0.9365 | 0.9315 | 0.9340 | 0.8703 |
| KNN | 0.9374 | 0.9770 | 0.9523 | 0.9188 | 0.9353 | 0.8752 |
| Naive Bayes | 0.7398 | 0.7966 | 0.6919 | 0.8503 | 0.7629 | 0.4939 |
| Support Vector Machine | 0.9543 | 0.9885 | 0.9540 | 0.9531 | 0.9535 | 0.9085 |

Settings: Logistic Regression `max_iter=2000`, Decision Tree `max_depth=10`, KNN `k=7`, Gaussian Naive Bayes with defaults, SVM with RBF kernel and `C=1.0`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline: accuracy 0.9405 and AUC 0.9813 after scaling. Errors are balanced (67 legitimate pages flagged as phishing, 67 phishing pages missed), so precision and recall are identical. It shows the classes are mostly linearly separable, and it is the fastest and most interpretable model. |
| Decision Tree | Accuracy 0.9352 and F1 0.9340, but the lowest AUC of the four good models (0.9274) because a tree outputs coarse probabilities. It missed 76 phishing pages and flagged 70 legitimate ones. Depth is limited to 10 to reduce overfitting. |
| KNN | Accuracy 0.9374 and the highest precision after SVM (0.9523), with only 51 false alarms, but the lowest recall of the four good models (0.9188): 90 phishing pages were missed. It stores the full training set, so the saved file is large and prediction is slower. |
| Naive Bayes | By far the weakest: accuracy 0.7398 and MCC 0.4939. Many features are strongly correlated (for example the URL length, hostname length and word-length features), which breaks the independence assumption. It flags 420 legitimate pages as phishing (low precision 0.6919) although recall is fairly high (0.8503). |
| Support Vector Machine | Best on every metric: accuracy 0.9543, AUC 0.9885, precision 0.9540, recall 0.9531, F1 0.9535 and MCC 0.9085. The RBF kernel captures non-linear interactions between URL, content and domain features that Logistic Regression cannot. Only 51 false alarms and 52 missed phishing pages. |
| Overall winner | **Support Vector Machine.** It has the best score on all six metrics. Logistic Regression is second and is a good choice when speed and interpretability matter. Naive Bayes is not suitable for this dataset. |

## e. Streamlit app

- **Sidebar:** upload a test CSV (or tick the bundled `test_data.csv`) and pick a model from the dropdown.
- **Evaluation tab:** the six metrics for the chosen model, a confusion matrix, a classification report and a download button for the predictions.
- **Model comparison tab:** a bar chart for any metric and a table with all metrics for all five models on the uploaded file.
- **Data preview tab:** class balance and the first 100 rows.

The uploaded CSV must contain the 87 numeric feature columns and the `status` column (`phishing` / `legitimate`). Extra columns and a different column order are fine. A missing column, empty cells, non-numeric features or unknown labels show an error message. The app only loads the saved models from `model/`; it never trains on the uploaded data.
