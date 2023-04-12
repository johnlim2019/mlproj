# 50.007 Machine Learning 2023 Spring Project

## Team members

- Alphonsus Tan
- Cheh Kang Ming
- John Lim

# Important folder structure
> Data folders should be placed in the root project directory.

    .
    ├── EN                      # Stores data files for language='EN'
    ├── FR                      # Stores data files for language='EN'
    ├── EvalScript              # Scripts for evaluating output
    ├── part1.py                # Script implementing Part 1
    ├── part2.py                # Script implementing Part 2
    ├── part3.py                # Script implementing Part 3
    ├── part4.py                # Script implementing Part 4
    ├── LICENSE
    └── README.md

> Data folders should contain a `train` file for training, a `dev.in` for testing. `dev.out` for evaluation is also required to perform calculations on the precision, recall and F scores on the test results. Example of data folder structure is as shown

    EN
    ├── dev.in
    ├── dev.out
    └── train

# Part 1: Emission Parameters

This section learns the emission parameters from `train` in the respective data folders and implements a simple sentiment analysis system.

Output is written to `dev.p1.out` in the respective data folders.

# Part 2: Transition Parameters and Viterbi

This section learns the transition parameters from `train` in the respective data folders and implements the viterbi algorithm under the simple first-order HMM assumption.

Output is written to `dev.p2.out` in the respective data folders.

# Part 3: Second-order HMM

This section learns the transition parameters for a second-order HMM model and implements the viterbi algorithm for decoding this second-order HMM.

Output is written to `dev.p3.out` in the respective data folders.

# Part 4: Design Challenge
Naive Bayes Model

We previously did not include any preprocessing on the input in the previous parts. As part of our BN model we try to change certain classes of words into a single class. Such as all punctuation-only tokens = #PUNC# observation. 

EN 
1. No preprocessing
   1. Entity  F: 0.6427
   2. Sentiment  F: 0.5148
2. lowercase
   1. Entity  F: 0.6457
   2. Sentiment  F: 0.5234
3. filter out stop words
   1. Entity  F: 0.6052
   2. Sentiment  F: 0.3596
4. filter out punctuation 
   1. Entity  F: 0.6383
   2. Sentiment  F: 0.5104
5. filter out links and usernames
   1. Entity  F: 0.6648
   2. Sentiment  F: 0.5325

FR
1. No preprocessing
   1. Entity  F: 0.6052
   2. Sentiment  F: 0.3596
2. lowercase
   1. Entity  F: 0.6052   
   2. Sentiment  F: 0.3596
3. filter out stop words
   1. Entity  F: 0.6052
   2. Sentiment  F: 0.3596
4. filter out punctuation 
   1. Entity  F: 0.6052
   2. Sentiment  F: 0.3596
5. filter out links and usernames
   1. Entity  F: 0.6052
   2. Sentiment  F: 0.3596


Final preprocessing choice is lowercase and filter links and usernames
EN 
#Entity in gold data: 802
#Entity in prediction: 1017

#Correct Entity : 609
Entity  precision: 0.5988
Entity  recall: 0.7594
Entity  F: 0.6696

#Correct Sentiment : 492
Sentiment  precision: 0.4838
Sentiment  recall: 0.6135
Sentiment  F: 0.5410

FR 
#Entity in gold data: 802
#Entity in prediction: 1128

#Correct Entity : 584
Entity  precision: 0.5177
Entity  recall: 0.7282
Entity  F: 0.6052

#Correct Sentiment : 347
Sentiment  precision: 0.3076
Sentiment  recall: 0.4327
Sentiment  F: 0.3596