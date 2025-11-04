===================================================================================================
HW2: Machine Learning for Sensing
Group 1 Members:
Nisa Defne Aksu
Barkın Var
Pelin Karadal
Shahd Sherif
===================================================================================================
How to Run:
Download the UCI HAR Dataset from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
Place the dataset in the same folder as the script (folder name: UCI HAR Dataset).
Run the script: python main.py
Install packages: pip install -r requirements.txt
Make sure you are in the same directory as your requirements.txt
==================================== PART 1 =======================================================
Aggregate KNN
Goal: Evaluate activity recognition across all users combined.
Method: 5-fold cross-validation was performed on the training set to determine the optimal number of neighbors (k), testing values from 1 to 20.
Training: The KNN classifier was trained on the aggregated data from all subjects using the best k.
Evaluation: The model was evaluated on the test set. Precision, recall, and F1-score were computed manually (without using built-in metric functions), along with the confusion matrix for detailed performance analysis.

User-specific KNN
Goal: Evaluate personalization by training a separate KNN model per individual user (30 users in total).
Method: Each user’s data was split into training and testing subsets (typically 80/20). Metrics (precision, recall, F1) were calculated manually for each user.
Parameter tuning: Since user-specific datasets are much smaller, cross-validation becomes unstable. The F1-scores can fluctuate considerably between runs. The best k was chosen manually by experimenting with different values and comparing the average F1-score across users to ensure stability and fairness.

Terminal Output:
===Part 1===

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

Best k = 16 (Mean F1 = 0.9012)

=== Aggregate KNN Classifier ===
Train: (7352, 561), Test: (2947, 561)

Training KNN with k = 16 ...
Predicting test data...

=== Manual Evaluation Metrics (Aggregate) ===
Precision (weighted): 0.911
Recall (weighted):    0.906
F1-score (weighted):  0.905

Confusion Matrix:
[[489   0   7   0   0   0]
 [ 39 426   6   0   0   0]
 [ 51  47 322   0   0   0]
 [  0   4   0 401  86   0]
 [  0   0   0  35 497   0]
 [  0   0   0   1   1 535]]

=== Running Individual User-Specific KNN Classifiers ===

Total users: 30

Printing both sklearn built-in metrics and manual calculation metrics for comparison. 

--- User 1 ---
Sklearn -> Precision: 1.000, Recall: 1.000, F1: 1.000
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 2 ---
Sklearn -> Precision: 0.970, Recall: 0.967, F1: 0.967
Manual  -> Precision: 0.970, Recall: 0.967, F1: 0.967

--- User 3 ---
Sklearn -> Precision: 0.962, Recall: 0.957, F1: 0.956
Manual  -> Precision: 0.962, Recall: 0.957, F1: 0.956

--- User 4 ---
Sklearn -> Precision: 0.946, Recall: 0.922, F1: 0.916
Manual  -> Precision: 0.946, Recall: 0.922, F1: 0.916

--- User 5 ---
Sklearn -> Precision: 0.952, Recall: 0.951, F1: 0.950
Manual  -> Precision: 0.952, Recall: 0.951, F1: 0.950

--- User 6 ---
Sklearn -> Precision: 0.955, Recall: 0.938, F1: 0.936
Manual  -> Precision: 0.955, Recall: 0.938, F1: 0.936

--- User 7 ---
Sklearn -> Precision: 0.973, Recall: 0.968, F1: 0.967
Manual  -> Precision: 0.973, Recall: 0.968, F1: 0.967

--- User 8 ---
Sklearn -> Precision: 1.000, Recall: 1.000, F1: 1.000
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 9 ---
Sklearn -> Precision: 0.934, Recall: 0.931, F1: 0.930
Manual  -> Precision: 0.934, Recall: 0.931, F1: 0.930

--- User 10 ---
Sklearn -> Precision: 1.000, Recall: 1.000, F1: 1.000
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 11 ---
Sklearn -> Precision: 0.974, Recall: 0.969, F1: 0.969
Manual  -> Precision: 0.974, Recall: 0.969, F1: 0.969

--- User 12 ---
Sklearn -> Precision: 0.986, Recall: 0.984, F1: 0.984
Manual  -> Precision: 0.986, Recall: 0.984, F1: 0.984

--- User 13 ---
Sklearn -> Precision: 0.986, Recall: 0.985, F1: 0.985
Manual  -> Precision: 0.986, Recall: 0.985, F1: 0.985

--- User 14 ---
Sklearn -> Precision: 0.947, Recall: 0.938, F1: 0.938
Manual  -> Precision: 0.947, Recall: 0.938, F1: 0.938

--- User 15 ---
Sklearn -> Precision: 0.986, Recall: 0.985, F1: 0.985
Manual  -> Precision: 0.986, Recall: 0.985, F1: 0.985

--- User 16 ---
Sklearn -> Precision: 0.941, Recall: 0.919, F1: 0.914
Manual  -> Precision: 0.941, Recall: 0.919, F1: 0.914

--- User 17 ---
Sklearn -> Precision: 0.976, Recall: 0.973, F1: 0.973
Manual  -> Precision: 0.976, Recall: 0.973, F1: 0.973

--- User 18 ---
Sklearn -> Precision: 0.960, Recall: 0.959, F1: 0.959
Manual  -> Precision: 0.960, Recall: 0.959, F1: 0.959

--- User 19 ---
Sklearn -> Precision: 0.987, Recall: 0.986, F1: 0.986
Manual  -> Precision: 0.987, Recall: 0.986, F1: 0.986

--- User 20 ---
Sklearn -> Precision: 0.987, Recall: 0.986, F1: 0.986
Manual  -> Precision: 0.987, Recall: 0.986, F1: 0.986

--- User 21 ---
Sklearn -> Precision: 0.964, Recall: 0.963, F1: 0.963
Manual  -> Precision: 0.964, Recall: 0.963, F1: 0.963

--- User 22 ---
Sklearn -> Precision: 0.986, Recall: 0.985, F1: 0.985
Manual  -> Precision: 0.986, Recall: 0.985, F1: 0.985

--- User 23 ---
Sklearn -> Precision: 0.977, Recall: 0.973, F1: 0.973
Manual  -> Precision: 0.977, Recall: 0.973, F1: 0.973

--- User 24 ---
Sklearn -> Precision: 0.988, Recall: 0.987, F1: 0.987
Manual  -> Precision: 0.988, Recall: 0.987, F1: 0.987

--- User 25 ---
Sklearn -> Precision: 0.952, Recall: 0.951, F1: 0.951
Manual  -> Precision: 0.952, Recall: 0.951, F1: 0.951

--- User 26 ---
Sklearn -> Precision: 0.975, Recall: 0.975, F1: 0.975
Manual  -> Precision: 0.975, Recall: 0.975, F1: 0.975

--- User 27 ---
Sklearn -> Precision: 0.988, Recall: 0.987, F1: 0.987
Manual  -> Precision: 0.988, Recall: 0.987, F1: 0.987

--- User 28 ---
Sklearn -> Precision: 0.924, Recall: 0.922, F1: 0.922
Manual  -> Precision: 0.924, Recall: 0.922, F1: 0.922

--- User 29 ---
Sklearn -> Precision: 0.975, Recall: 0.971, F1: 0.971
Manual  -> Precision: 0.975, Recall: 0.971, F1: 0.971

--- User 30 ---
Sklearn -> Precision: 0.977, Recall: 0.974, F1: 0.974
Manual  -> Precision: 0.977, Recall: 0.974, F1: 0.974

=== Summary Per User ===
    User  Precision    Recall        F1
0      1   1.000000  1.000000  1.000000
1      2   0.970113  0.967213  0.967100
2      3   0.962467  0.956522  0.956430
3      4   0.946289  0.921875  0.916088
4      5   0.951503  0.950820  0.950442
5      6   0.954872  0.938462  0.936358
6      7   0.972705  0.967742  0.967294
7      8   1.000000  1.000000  1.000000
8      9   0.934169  0.931034  0.929700
9     10   1.000000  1.000000  1.000000
10    11   0.974432  0.968750  0.968750
11    12   0.985577  0.984375  0.984276
12    13   0.986226  0.984848  0.984848
13    14   0.947253  0.938462  0.937581
14    15   0.986014  0.984848  0.984791
15    16   0.941032  0.918919  0.914264
16    17   0.975976  0.972973  0.972642
17    18   0.960209  0.958904  0.959089
18    19   0.987374  0.986111  0.985979
19    20   0.986796  0.985915  0.985861
20    21   0.964052  0.963415  0.963415
21    22   0.985714  0.984615  0.984593
22    23   0.976667  0.973333  0.973197
23    24   0.987879  0.987013  0.986996
24    25   0.951982  0.951220  0.950721
25    26   0.974684  0.974684  0.974684
26    27   0.987616  0.986842  0.986798
27    28   0.923864  0.922078  0.921978
28    29   0.974879  0.971014  0.970732
29    30   0.977489  0.974026  0.973773

Average F1 across users: 0.966
==================================== PART 2 =======================================================




==================================== PART 3 =======================================================
Implemented three perceptrons: OR, NAND, AND.
Combined them to implement XOR logic: XOR(A, B) = AND(OR(A,B), NAND(A,B))
Used a step function as the activation function.
Trained each perceptron using the perceptron learning rule.
All outputs are correct, meaning our perceptrons correctly implement XOR.

Terminal Output:
===Part 3===

Input A | Input B | XOR Output
-----------------------------
   0    |    0    |     0
   0    |    1    |     1
   1    |    0    |     1
   1    |    1    |     0
