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
Method: 5-fold cross-validation was performed on the training set to determine the optimal number of neighbors (k), testing values from 1 to 10.
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

Finding best k using validation fold:
```
k =  1 → Mean F1-score = 0.8739
k =  2 → Mean F1-score = 0.8635
k =  3 → Mean F1-score = 0.8858
k =  4 → Mean F1-score = 0.8852
k =  5 → Mean F1-score = 0.8893
k =  6 → Mean F1-score = 0.8889
k =  7 → Mean F1-score = 0.8937
k =  8 → Mean F1-score = 0.8935
k =  9 → Mean F1-score = 0.8929
k = 10 → Mean F1-score = 0.8922
```

Picked best k = 7 (Validation F1 = 0.8937)
<br>
=== Manual KNN Classifier For Aggregate Data ===
<br>
Train: (7352, 561), Test: (2947, 561)
<br>
Training manual KNN with k = 7...

<br>

=== Manual Evaluation Metrics (Aggregate) ===
<br>
Precision (weighted): 0.908
<br>
Recall (weighted):    0.903
<br>
F1-score (weighted):  0.903

<br>

=== Confusion Matrix ===
```
     1    2    3    4    5    6
1  482    3   11    0    0    0
2   42  423    6    0    0    0
3   48   40  332    0    0    0
4    0    4    0  394   93    0
5    0    0    0   35  497    0
6    0    0    0    2    1  534
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

Model: "sequential"
<br>
| Layer (type)                         | Output Shape                |         Param # |
|-------------------------------------:|----------------------------:|----------------:|
| conv1d (Conv1D)                      | (None, 126, 64)             |           1,792 |
| conv1d_1 (Conv1D)                    | (None, 124, 64)             |          12,352 |
| max_pooling1d (MaxPooling1D)         | (None, 62, 64)              |               0 |
| dropout (Dropout)                    | (None, 62, 64)              |               0 |
| flatten (Flatten)                    | (None, 3968)                |               0 |
| dense (Dense)                        | (None, 100)                 |         396,900 |
| dense_1 (Dense)                      | (None, 6)                   |             606 |

 Total params: 411,650 (1.57 MB)
 <br>
 Trainable params: 411,650 (1.57 MB)
 <br>
 Non-trainable params: 0 (0.00 B)

<br>

=== Manual Evaluation Metrics (Aggregate DNN) ===
<br>
Precision (weighted): 0.914
<br>
Recall (weighted):    0.913
<br>
F1-score (weighted):  0.913

<br>

=== Confusion Matrix (Labels 0-5) ===
<br>
```
     0    1    2    3    4    5
0  476    0   20    0    0    0
1   16  433   22    0    0    0
2    0    1  419    0    0    0
3    0    2    0  396   88    5
4    0    1    0   64  467    0
5    0   36    0    0    0  501
```

=== Running Individual User-Specific 1D-CNN Classifiers ===

Total users: 30

--- User 1 ---
<br>
Training 1D-CNN for user 1 (277 samples)...
<br>
Manual  -> Precision: 0.927, Recall: 0.900, F1: 0.881

--- User 2 ---
<br>
Training 1D-CNN for user 2 (241 samples)...
<br>
Manual  -> Precision: 0.955, Recall: 0.951, F1: 0.951

--- User 3 ---
<br>
Training 1D-CNN for user 3 (272 samples)...
<br>
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 4 ---
<br>
Training 1D-CNN for user 4 (253 samples)...
<br>
Manual  -> Precision: 0.939, Recall: 0.922, F1: 0.921

--- User 5 ---
<br>
Training 1D-CNN for user 5 (241 samples)...
<br>
Manual  -> Precision: 0.945, Recall: 0.934, F1: 0.933

--- User 6 ---
<br>
Training 1D-CNN for user 6 (260 samples)...
<br>
Manual  -> Precision: 0.923, Recall: 0.892, F1: 0.888

--- User 7 ---
<br>
Training 1D-CNN for user 7 (246 samples)...
<br>
Manual  -> Precision: 0.906, Recall: 0.903, F1: 0.899

--- User 8 ---
<br>
Training 1D-CNN for user 8 (224 samples)...
<br>
Manual  -> Precision: 0.839, Recall: 0.842, F1: 0.838

--- User 9 ---
<br>
Training 1D-CNN for user 9 (230 samples)...
<br>
Manual  -> Precision: 0.797, Recall: 0.759, F1: 0.723

--- User 10 ---
<br>
Training 1D-CNN for user 10 (235 samples)...
<br>
Manual  -> Precision: 0.819, Recall: 0.797, F1: 0.787

--- User 11 ---
<br>
Training 1D-CNN for user 11 (252 samples)...
<br>
Manual  -> Precision: 0.986, Recall: 0.984, F1: 0.984

--- User 12 ---
<br>
Training 1D-CNN for user 12 (256 samples)...
<br>
Manual  -> Precision: 0.788, Recall: 0.766, F1: 0.771

--- User 13 ---
<br>
Training 1D-CNN for user 13 (261 samples)...
<br>
Manual  -> Precision: 0.907, Recall: 0.894, F1: 0.896

--- User 14 ---
<br>
Training 1D-CNN for user 14 (258 samples)...
<br>
Manual  -> Precision: 0.946, Recall: 0.938, F1: 0.937

--- User 15 ---
<br>
Training 1D-CNN for user 15 (262 samples)...
<br>
Manual  -> Precision: 0.974, Recall: 0.970, F1: 0.969

--- User 16 ---
<br>
Training 1D-CNN for user 16 (292 samples)...
<br>
Manual  -> Precision: 0.936, Recall: 0.919, F1: 0.916

--- User 17 ---
<br>
Training 1D-CNN for user 17 (294 samples)...
<br>
Manual  -> Precision: 0.946, Recall: 0.919, F1: 0.907

--- User 18 ---
<br>
Training 1D-CNN for user 18 (291 samples)...
<br>
Manual  -> Precision: 0.960, Recall: 0.945, F1: 0.946

--- User 19 ---
<br>
Training 1D-CNN for user 19 (288 samples)...
<br>
Manual  -> Precision: 0.972, Recall: 0.972, F1: 0.972

--- User 20 ---
<br>
Training 1D-CNN for user 20 (283 samples)...
<br>
Manual  -> Precision: 0.887, Recall: 0.887, F1: 0.881

--- User 21 ---
<br>
Training 1D-CNN for user 21 (326 samples)...
<br>
Manual  -> Precision: 0.979, Recall: 0.976, F1: 0.975

--- User 22 ---
<br>
Training 1D-CNN for user 22 (256 samples)...
<br>
Manual  -> Precision: 0.928, Recall: 0.923, F1: 0.925

--- User 23 ---
<br>
Training 1D-CNN for user 23 (297 samples)...
<br>
Manual  -> Precision: 0.864, Recall: 0.840, F1: 0.834

--- User 24 ---
<br>
Training 1D-CNN for user 24 (304 samples)...
<br>
Manual  -> Precision: 0.935, Recall: 0.935, F1: 0.935

--- User 25 ---
<br>
Training 1D-CNN for user 25 (327 samples)...
<br>
Manual  -> Precision: 0.971, Recall: 0.963, F1: 0.964

--- User 26 ---
<br>
Training 1D-CNN for user 26 (313 samples)...
<br>
Manual  -> Precision: 0.955, Recall: 0.937, F1: 0.936

--- User 27 ---
<br>
Training 1D-CNN for user 27 (300 samples)...
<br>
Manual  -> Precision: 0.962, Recall: 0.961, F1: 0.961

--- User 28 ---
<br>
Training 1D-CNN for user 28 (305 samples)...
<br>
Manual  -> Precision: 0.821, Recall: 0.779, F1: 0.767

--- User 29 ---
<br>
Training 1D-CNN for user 29 (275 samples)...
<br>
Manual  -> Precision: 1.000, Recall: 1.000, F1: 1.000

--- User 30 ---
<br>
Training 1D-CNN for user 30 (306 samples)...
<br>
Manual  -> Precision: 0.988, Recall: 0.987, F1: 0.987

=== Summary Per User (DNN) ===
|   User|  Precision|    Recall|        F1|
|------:|----------:|---------:|---------:|
|0      1|   0.926923|  0.900000|  0.880855
|1      2|   0.955359|  0.950820|  0.950620
|2      3|   1.000000|  1.000000|  1.000000
|3      4|   0.938702|  0.921875|  0.920823
|4      5|   0.944515|  0.934426|  0.933288
|5      6|   0.923177|  0.892308|  0.888163
|6      7|   0.905844|  0.903226|  0.899267
|7      8|   0.839181|  0.842105|  0.838460
|8      9|   0.796503|  0.758621|  0.722504
|9     10|   0.819058|  0.796610|  0.787121
|10    11|   0.985938|  0.984375|  0.984447
|11    12|   0.787527|  0.765625|  0.770881
|12    13|   0.906866|  0.893939|  0.895502
|13    14|   0.945661|  0.938462|  0.937064
|14    15|   0.974359|  0.969697|  0.968795
|15    16|   0.936235|  0.918919|  0.916106
|16    17|   0.945946|  0.918919|  0.906757
|17    18|   0.959817|  0.945205|  0.945878
|18    19|   0.972222|  0.972222|  0.972222
|19    20|   0.887431|  0.887324|  0.880617
|20    21|   0.979362|  0.975610|  0.975102
|21    22|   0.927692|  0.923077|  0.924509
|22    23|   0.864470|  0.840000|  0.834418
|23    24|   0.935465|  0.935065|  0.934818
|24    25|   0.971254|  0.963415|  0.963577
|25    26|   0.955324|  0.936709|  0.935534
|26    27|   0.961722|  0.960526|  0.960608
|27    28|   0.821299|  0.779221|  0.767003
|28    29|   1.000000|  1.000000|  1.000000
|29    30|   0.987941|  0.987013|  0.986994

Average F1 across users: 0.909



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

