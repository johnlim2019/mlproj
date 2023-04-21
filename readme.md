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

# Instructions to run

## Dev mode

The default mode. This is used when your data folder follows the above shown structure. The models will be trained on `train`, and tested on `dev.in`. The name of the output file will be dependant on the script that is run. To run in dev mode, you only need to specify `lang`, the name of your data folder. Example as shown.

```
python3 part1.py <lang path> <mode=dev (default, optional)>
```

## Test mode

This is used if you wish to specify the **exact** path of the input test file and the **exact** output result file path. You still need to input the path of `lang` where the train file for training will be used. Example as shown.

```
python3 part1.py EN test <exact input path> <exact output path>
```

# Part 1: Emission Parameters

This section learns the emission parameters from `train` in the respective data folders and implements a simple sentiment analysis system.

To run the code, navigate to the root project directory and input the following command, specifying the language (EN or FR) to be used:
```console
python3 part1.py {language}

EN
#Entity in gold data: 802
#Entity in prediction: 1148

#Correct Entity : 614
Entity  precision: 0.5348
Entity  recall: 0.7656
Entity  F: 0.6297

#Correct Sentiment : 448
Sentiment  precision: 0.3902
Sentiment  recall: 0.5586
Sentiment  F: 0.4595

FR
#Entity in gold data: 238
#Entity in prediction: 1114

#Correct Entity : 186
Entity  precision: 0.1670
Entity  recall: 0.7815
Entity  F: 0.2751

#Correct Sentiment : 79
Sentiment  precision: 0.0709
Sentiment  recall: 0.3319
Sentiment  F: 0.1169
```

Output is written to `dev.p1.out` in the respective data folders.

# Part 2: Transition Parameters and Viterbi

This section learns the transition parameters from `train` in the respective data folders and implements the viterbi algorithm under the simple first-order HMM assumption.

To run the code, navigate to the root project directory and input the following command, specifying the language (EN or FR) to be used:
```console
python3 part2.py {language}

EN
#Entity in gold data: 802
#Entity in prediction: 821

#Correct Entity : 537
Entity  precision: 0.6541
Entity  recall: 0.6696
Entity  F: 0.6617

#Correct Sentiment : 470
Sentiment  precision: 0.5725
Sentiment  recall: 0.5860
Sentiment  F: 0.5792

FR
#Entity in gold data: 238
#Entity in prediction: 445

#Correct Entity : 136
Entity  precision: 0.3056
Entity  recall: 0.5714
Entity  F: 0.3982

#Correct Sentiment : 76
Sentiment  precision: 0.1708
Sentiment  recall: 0.3193
Sentiment  F: 0.2225
```
Output is written to `dev.p2.out` in the respective data folders.

# Part 3: Second-order HMM

This section learns the transition parameters for a second-order HMM model and implements the viterbi algorithm for decoding this second-order HMM.

To run the code, navigate to the root project directory and input the following command, specifying the language (EN or FR) to be used:
```console
python3 part3.py {language}

EN
#Entity in gold data: 802
#Entity in prediction: 697

#Correct Entity : 450
Entity  precision: 0.6456
Entity  recall: 0.5611
Entity  F: 0.6004

#Correct Sentiment : 391
Sentiment  precision: 0.5610
Sentiment  recall: 0.4875
Sentiment  F: 0.5217

FR
#Entity in gold data: 238
#Entity in prediction: 292

#Correct Entity : 133
Entity  precision: 0.4555
Entity  recall: 0.5588
Entity  F: 0.5019

#Correct Sentiment : 76
Sentiment  precision: 0.2603
Sentiment  recall: 0.3193
Sentiment  F: 0.2868
```

Output is written to `dev.p3.out` in the respective data folders.

# Part 4: Design Challenge
Data Preprocessing

We previously did not include any preprocessing on the input in the previous parts. As part of our design challenge,  we preprocessed our training data by changing our results to lowercase and tagged links and usernames to #LNK# tag. We applied the processing step to our first order Hidden Markov Model as it currently yields the best F score.


```console
python3 part4.py {language}

EN 
#Entity in gold data: 802
#Entity in prediction: 807

#Correct Entity : 547
Entity  precision: 0.6778
Entity  recall: 0.6820
Entity  F: 0.6799

#Correct Sentiment : 486
Sentiment  precision: 0.6022
Sentiment  recall: 0.6060
Sentiment  F: 0.6041

FR
#Entity in gold data: 238
#Entity in prediction: 424

#Correct Entity : 133
Entity  precision: 0.3137
Entity  recall: 0.5588
Entity  F: 0.4018

#Correct Sentiment : 82
Sentiment  precision: 0.1934
Sentiment  recall: 0.3445
Sentiment  F: 0.2477
```

Output is written to `dev.p4.out` in the respective data folders.