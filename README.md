# Development and Validation of an Interpretable Machine Learning Model for Predicting Pathological Complete Response and Prognostic Stratification Following Neoadjuvant Immunochemotherapy for NSCLC: A Multicenter Retrospective Study
Accurately predicting pathological complete response (pCR) prior to neoadjuvant immunochemotherapy (NICT) is crucial for the treatment of patients with resectable non-small cell lung cancer (NSCLC). In this study, we developed a multimodal integration pipeline that combines traditional tabular features with both semantic embeddings and radiological and pathological report embeddings from MC-BERT. This approach shows promise for supporting clinical decision-making in resectable NSCLC by helping to avoid overtreatment, advancing personalized and precision cancer therapy, and providing valuable insights into patient survival.

### Extracting text and semantic features using MC-BERT

python emb.py

MC-BERT: https://huggingface.co/freedomking/mc-bert

### Numerical feature selection

LASSO.ipynb

### Train and test

model.ipynb
