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

