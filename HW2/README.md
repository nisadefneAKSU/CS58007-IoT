======================================================================
<br>
HW2: Machine Learning for Sensing
<br>
Group 1 Members:
* Nisa Defne Aksu
* Barkın Var
* Pelin Karadal
* Shahd Sherif

======================================================================
<br>
How to Run:
<br>
Download the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
<br>
Place the dataset in the same folder as the script (folder name: UCI HAR Dataset).
<br>
Run the script: `python main.py`
<br>
Install packages: `pip install -r requirements.txt`
<br>
Make sure you are in the same directory as your requirements.txt file
<br>
==============================PART 1====================================
<br>
Aggregate KNN:
<br>
Goal: Evaluate activity recognition across all users combined.
<br>
Method: 5-fold cross-validation was performed on the training set to determine the optimal number of neighbors (k), testing values from 1 to 20.
<br>
Training: The KNN classifier was trained on the aggregated data from all subjects using the best k.
<br>
Evaluation: The model was evaluated on the test set. Precision, recall, and F1-score were computed manually (without using built-in metric functions), along with the confusion matrix for detailed performance analysis.

<br>

User-specific KNN:
<br>
Goal: Evaluate personalization by training a separate KNN model per individual user (30 users in total).
<br>
Method: Each user’s data was split into training and testing subsets (typically 80/20). Metrics (precision, recall, F1) were calculated manually for each user.
<br>
Parameter tuning: Since user-specific datasets are much smaller, cross-validation becomes unstable. The F1-scores can fluctuate considerably between runs. The best k was chosen manually by experimenting with different values and comparing the average F1-score across users to ensure stability and fairness.

<br>

Terminal Output:
<br>
===Part 1===
```
k =  1 → Mean F1-score = 0.8783
k =  2 → Mean F1-score = 0.8668
k =  3 → Mean F1-score = 0.8910
k =  4 → Mean F1-score = 0.8897
k =  5 → Mean F1-score = 0.8958
k =  6 → Mean F1-score = 0.8946
k =  7 → Mean F1-score = 0.8988
k =  8 → Mean F1-score = 0.8978
k =  9 → Mean F1-score = 0.8973
k = 10 → Mean F1-score = 0.8964
k = 11 → Mean F1-score = 0.8977
k = 12 → Mean F1-score = 0.8976
k = 13 → Mean F1-score = 0.8987
k = 14 → Mean F1-score = 0.9002
k = 15 → Mean F1-score = 0.9009
k = 16 → Mean F1-score = 0.9012
k = 17 → Mean F1-score = 0.9010
k = 18 → Mean F1-score = 0.8999
k = 19 → Mean F1-score = 0.8993
k = 20 → Mean F1-score = 0.8996
```

Best k = 16 (Mean F1 = 0.9012)
<br>
=== Aggregate KNN Classifier ===
<br>
Train: (7352, 561), Test: (2947, 561)
<br>
Training KNN with k = 16 ...
<br>
Predicting test data...

<br>

=== Manual Evaluation Metrics (Aggregate) ===
<br>
Precision (weighted): 0.911
<br>
Recall (weighted):    0.906
<br>
F1-score (weighted):  0.905

<br>

//Remove this Classification Report???

=== Classification Report ===
              precision    recall  f1-score   support

           1     0.8446    0.9859    0.9098       496
           2     0.8931    0.9045    0.8987       471
           3     0.9612    0.7667    0.8530       420
           4     0.9176    0.8167    0.8642       491
           5     0.8510    0.9342    0.8907       532
           6     1.0000    0.9963    0.9981       537

    accuracy                         0.9060      2947
   macro avg     0.9112    0.9007    0.9024      2947
weighted avg     0.9106    0.9060    0.9050      2947

=== Confusion Matrix ===
```
[[489   0   7   0   0   0]
 [ 39 426   6   0   0   0]
 [ 51  47 322   0   0   0]
 [  0   4   0 401  86   0]
 [  0   0   0  35 497   0]
 [  0   0   0   1   1 535]]
``` 

=== Running Individual User-Specific KNN Classifiers ===

Total users: 30

--- User 1 ---
<br>
Sklearn -> Precision: 1.000, Recall: 1.000, F1: 1.000
<br>
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 2 ---
<br>
Sklearn -> Precision: 0.970, Recall: 0.967, F1: 0.967
<br>
Manual  -> Precision: 0.970, Recall: 0.967, F1: 0.967

--- User 3 ---
<br>
Sklearn -> Precision: 0.962, Recall: 0.957, F1: 0.956
<br>
Manual  -> Precision: 0.962, Recall: 0.957, F1: 0.956

--- User 4 ---
<br>
Sklearn -> Precision: 0.946, Recall: 0.922, F1: 0.916
<br>
Manual  -> Precision: 0.946, Recall: 0.922, F1: 0.916

--- User 5 ---
<br>
Sklearn -> Precision: 0.952, Recall: 0.951, F1: 0.950
<br>
Manual  -> Precision: 0.952, Recall: 0.951, F1: 0.950

--- User 6 ---
<br>
Sklearn -> Precision: 0.955, Recall: 0.938, F1: 0.936
<br>
Manual  -> Precision: 0.955, Recall: 0.938, F1: 0.936

--- User 7 ---
<br>
Sklearn -> Precision: 0.973, Recall: 0.968, F1: 0.967
<br>
Manual  -> Precision: 0.973, Recall: 0.968, F1: 0.967

--- User 8 ---
<br>
Sklearn -> Precision: 1.000, Recall: 1.000, F1: 1.000
<br>
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 9 ---
<br>
Sklearn -> Precision: 0.934, Recall: 0.931, F1: 0.930
<br>
Manual  -> Precision: 0.934, Recall: 0.931, F1: 0.930

--- User 10 ---
<br>
Sklearn -> Precision: 1.000, Recall: 1.000, F1: 1.000
<br>
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 11 ---
<br>
Sklearn -> Precision: 0.974, Recall: 0.969, F1: 0.969
<br>
Manual  -> Precision: 0.974, Recall: 0.969, F1: 0.969

--- User 12 ---
<br>
Sklearn -> Precision: 0.986, Recall: 0.984, F1: 0.984
<br>
Manual  -> Precision: 0.986, Recall: 0.984, F1: 0.984

--- User 13 ---
<br>
Sklearn -> Precision: 0.986, Recall: 0.985, F1: 0.985
<br>
Manual  -> Precision: 0.986, Recall: 0.985, F1: 0.985

--- User 14 ---
<br>
Sklearn -> Precision: 0.947, Recall: 0.938, F1: 0.938
<br>
Manual  -> Precision: 0.947, Recall: 0.938, F1: 0.938

--- User 15 ---
<br>
Sklearn -> Precision: 0.986, Recall: 0.985, F1: 0.985
<br>
Manual  -> Precision: 0.986, Recall: 0.985, F1: 0.985

--- User 16 ---
<br>
Sklearn -> Precision: 0.941, Recall: 0.919, F1: 0.914
<br>
Manual  -> Precision: 0.941, Recall: 0.919, F1: 0.914

--- User 17 ---
<br>
Sklearn -> Precision: 0.976, Recall: 0.973, F1: 0.973
<br>
Manual  -> Precision: 0.976, Recall: 0.973, F1: 0.973

--- User 18 ---
<br>
Sklearn -> Precision: 0.960, Recall: 0.959, F1: 0.959
<br>
Manual  -> Precision: 0.960, Recall: 0.959, F1: 0.959

--- User 19 ---
<br>
Sklearn -> Precision: 0.987, Recall: 0.986, F1: 0.986
<br>
Manual  -> Precision: 0.987, Recall: 0.986, F1: 0.986

--- User 20 ---
<br>
Sklearn -> Precision: 0.987, Recall: 0.986, F1: 0.986
<br>
Manual  -> Precision: 0.987, Recall: 0.986, F1: 0.986

--- User 21 ---
<br>
Sklearn -> Precision: 0.964, Recall: 0.963, F1: 0.963
<br>
Manual  -> Precision: 0.964, Recall: 0.963, F1: 0.963

--- User 22 ---
<br>
Sklearn -> Precision: 0.986, Recall: 0.985, F1: 0.985
<br>
Manual  -> Precision: 0.986, Recall: 0.985, F1: 0.985

--- User 23 ---
<br>
Sklearn -> Precision: 0.977, Recall: 0.973, F1: 0.973
<br>
Manual  -> Precision: 0.977, Recall: 0.973, F1: 0.973

--- User 24 ---
<br>
Sklearn -> Precision: 0.988, Recall: 0.987, F1: 0.987
<br>
Manual  -> Precision: 0.988, Recall: 0.987, F1: 0.987

--- User 25 ---
<br>
Sklearn -> Precision: 0.952, Recall: 0.951, F1: 0.951
<br>
Manual  -> Precision: 0.952, Recall: 0.951, F1: 0.951

--- User 26 ---
<br>
Sklearn -> Precision: 0.975, Recall: 0.975, F1: 0.975
<br>
Manual  -> Precision: 0.975, Recall: 0.975, F1: 0.975

--- User 27 ---
<br>
Sklearn -> Precision: 0.988, Recall: 0.987, F1: 0.987
<br>
Manual  -> Precision: 0.988, Recall: 0.987, F1: 0.987

--- User 28 ---
<br>
Sklearn -> Precision: 0.924, Recall: 0.922, F1: 0.922
<br>
Manual  -> Precision: 0.924, Recall: 0.922, F1: 0.922

--- User 29 ---
<br>
Sklearn -> Precision: 0.975, Recall: 0.971, F1: 0.971
<br>
Manual  -> Precision: 0.975, Recall: 0.971, F1: 0.971

--- User 30 ---
<br>
Sklearn -> Precision: 0.977, Recall: 0.974, F1: 0.974
<br>
Manual  -> Precision: 0.977, Recall: 0.974, F1: 0.974

<br>

=== Summary Per User ===
|   | User | Precision | Recall | F1 |
|----:|----------:|----------:|----------:|----------:|
|0   |   1  | 1.000000 | 1.000000 | 1.000000|
|1   |   2  | 0.970113 | 0.967213 | 0.967100|
|2   |   3  | 0.962467 | 0.956522 | 0.956430|
|3   |   4  | 0.946289 | 0.921875 | 0.916088|
|4   |   5  | 0.951503 | 0.950820 | 0.950442|
|5   |   6  | 0.954872 | 0.938462 | 0.936358|
|6   |   7  | 0.972705 | 0.967742 | 0.967294|
|7   |   8  | 1.000000 | 1.000000 | 1.000000|
|8   |   9  | 0.934169 | 0.931034 | 0.929700|
|9   |  10  | 1.000000 | 1.000000 | 1.000000|
|10  |  11  | 0.974432 | 0.968750 | 0.968750|
|11  |  12  | 0.985577 | 0.984375 | 0.984276|
|12  |  13  | 0.986226 | 0.984848 | 0.984848|
|13  |  14  | 0.947253 | 0.938462 | 0.937581|
|14  |  15  | 0.986014 | 0.984848 | 0.984791|
|15  |  16  | 0.941032 | 0.918919 | 0.914264|
|16  |  17  | 0.975976 | 0.972973 | 0.972642|
|17  |  18  | 0.960209 | 0.958904 | 0.959089|
|18  |  19  | 0.987374 | 0.986111 | 0.985979|
|19  |  20  | 0.986796 | 0.985915 | 0.985861|
|20  |  21  | 0.964052 | 0.963415 | 0.963415|
|21  |  22  | 0.985714 | 0.984615 | 0.984593|
|22  |  23  | 0.976667 | 0.973333 | 0.973197|
|23  |  24  | 0.987879 | 0.987013 | 0.986996|
|24  |  25  | 0.951982 | 0.951220 | 0.950721|
|25  |  26  | 0.974684 | 0.974684 | 0.974684|
|26  |  27  | 0.987616 | 0.986842 | 0.986798|
|27  |  28  | 0.923864 | 0.922078 | 0.921978|
|28  |  29  | 0.974879 | 0.971014 | 0.970732|
|29  |  30  | 0.977489 | 0.974026 | 0.973773|

<br>

Average F1 across users: 0.966
<br>
==============================PART 2====================================
<br>
<br>
<br>
<br>
==============================PART 3====================================

Implemented three perceptrons: OR, NAND, AND.
<br>
Combined them to implement XOR logic: XOR(A, B) = AND(OR(A,B), NAND(A,B))
<br>
Used a step function as the activation function.
<br>
Trained each perceptron using the perceptron learning rule.
<br>
All outputs are correct, meaning our perceptrons correctly implement XOR.

<br>

Terminal Output:
<br>
===Part 3===

Input A | Input B | XOR Output
--------|---------|------------|
   0    |    0    |     0
   0    |    1    |     1
   1    |    0    |     1
   1    |    1    |     0

