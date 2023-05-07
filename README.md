# Perfectly_Fair_Differentially_Private

The proposed selection procedures applied on two different datasets

## FICO Credit Scores

We investigated our proposed methods on the following two different experiment settings on FICO credit score dataset

### 1 ) White vs. Black
### 2 ) White + Hispanic vs. Asian

#### File description

Since the two experimet settings are similar we explain each file in these two experiments together

- In "generate_sample.py" generates 100K samples and save samples in "checking_{wb\wha}_generated_samples_corrected.csv".


- In "{wb\wha}_debiased_multiple_selection.py", we applied oblivious_selection_procedure (similar $\epsilon$ applied on both groups) on debiased scores. The debiased model trained on the "checking_{wb\wha}_generated_samples_corrected.csv" using the adversarial debiasing method. We adopted the AdversarialDebiasing from [FairAI360](https://github.com/Trusted-AI/AIF360) Github repository and replaced the "adversarial_debiasing.py" file with ours which is available in ["adversarial_debiasing.py"](https://github.com/FairPrivate/Perfectly_Fair_Differentially_Private/blob/main/adversarial_debiasing.py). 
In the selection procedure we selected different number of applicants $m=\\{1, 2, 3, 4\\}$ from an applicant pool of $n=10$ people.

- In "{wb\wha}_noisy_selection.py", we applied the proposed selection procedures to select one applicant from applicants pool of 10 people for 10,000 times over different values of epsilon ranging $\epsilon\in[0,20]$

- In "{wb\wha}_noisy_multiple_selection.py", we applied the proposed selection procedures to select $m=\\{1, 2, 3, 4\\}$ applicant from applicants pool of 10 people for 10,000 times over different values of epsilon ranging $\epsilon\in[0,20]$

- In "{wb\wha}_single_selection_fairness_accuracy_figure.py", we tried to generate plots of single applicant selection of our proposed methods.

- In "{wb\wha}_multiple_selection_fairness_accuracy_figure.py", we tried to generate plots of multiple applicants selection of our proposed methods.

## Adult Income 

We investigated our proposed methods on the following experiment setting on Adult income dataset

### 1) White vs. Black

#### File description

- In "generate_qualification_scores.py", we tried to train an XGBoost model on adult dataset to assign a qualification score to each individual in the dataset.

- In "selection_procedure.py", we tried to apply our proposed methods on an applicant pool of 10 samples for 10,000 times over different values of epsilon ranging $\epsilon\in[0,20]$. Where, each sample has the following format $(r(X_i), A \in \\{White, Black\\}, Y \in \\{0, 1\\})$. We tried to select different number of applicants $m$ where $m=\\{1, 2, 3, 4\\}$. 

- In "selection_on_debiased_scores.py", we applied oblivious_selection_procedure (similar \epsilon is applied on both groups) on debiased scores. 

- In "single_selection_figure.py" and "multiple_selection_figure.py" we plot the results from the previous procedures.

## File description

In the ["adversarial_debiasing.py"](https://github.com/FairPrivate/Perfectly_Fair_Differentially_Private/blob/main/adversarial_debiasing.py) we have changed the procedure for training the adversary so that it satisfies equal opportunity fairness notion. To do so, we adopted the code from ["adversarial_debiasing.py"](https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/inprocessing/adversarial_debiasing.py) in [FairAI360](https://github.com/Trusted-AI/AIF360) Github repository. Since we wanted to satisfy equal opportunity according to [[1]](#1) we limited the adversary to only be trained on the qualified applicants.


## References
<a id="1">[1]</a> 
Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. 
Mitigatingunwanted biases with adversarial learning. 
In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pages 335–340, 2018.

