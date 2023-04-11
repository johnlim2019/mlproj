from part1 import generate_emission_matrix
import pandas as pd
from tqdm import tqdm
import sys


def get_emission(word, tag, emission_matrix, observed_words, tag_list):
    """Returns the emission probability of a word given a tag.

    Args:
        word (str): The word.
        tag (str): The tag.
        emission_matrix (dict): The emission matrix.
        observed_words (list): The list of observed words during training.
        tag_list (list): The list of tags observed during training.

    Returns:
        float: The emission probability of the word given the tag.
    """
    if tag not in tag_list:
        return 0
    if word in observed_words:
        return emission_matrix[tag_list.index(tag)][observed_words.index(word)]
    else:
        return emission_matrix[tag_list.index(tag)][observed_words.index('#UNK#')]


def get_unique_state_tuples(states):
    """Given a set of unique states, add all possible state tuples to a list.

    Args:
        states (set): The set of unique states. 

    Returns:
        list: The list of state tuples.
    """
    res = set()
    for i in states:
        for j in states:
            res.add((i, j))

    return list(res)


def fit_transitions_second_order(train_path):
    """Considers transitions of the form (t_i-2, t_i-1, t_i) as (t, u, v) and returns a dictionary of the form {(t, u, v): count}

    Args:
        train_path (str): The path to the training data.

    Returns:
        dict: {(t, u, v): count)}
        set: The set of unique states.
    """

    transitions_second_order = {}
    unique_states = set()
    unique_states.add('STOP')
    with open(train_path, 'r') as f:
        t = 'PRE_START'
        u = 'START'
        unique_states.add(t)
        unique_states.add(u)
        for line in f.readlines():
            split = line.strip().rsplit(maxsplit=1)
            # If the line is empty, we have reached the end of a sentence
            if len(split) < 2:
                transitions_second_order[(t, u, 'STOP')] = transitions_second_order.get(
                    (t, u, 'STOP'), 0) + 1
                # Reset the values of t and u
                t = 'PRE_START'
                u = 'START'
            else:
                v = split[1]
                unique_states.add(v)
                transitions_second_order[(t, u, v)] = transitions_second_order.get(
                    (t, u, v), 0) + 1
                t = u
                u = v

        transitions_first_order = {}
        for state_triplet, count in transitions_second_order.items():
            # Count (t -> u)
            transitions_first_order[(state_triplet[0], state_triplet[1])] = transitions_first_order.get(
                (state_triplet[0], state_triplet[1]), 0) + count

        # Compute final second order transitions
        # P(v | t, u) = P(t, u, v) / P(t, u)
        for state_triplet, count in transitions_second_order.items():
            transitions_second_order[state_triplet] = count / \
                transitions_first_order[(state_triplet[0], state_triplet[1])]

        return transitions_second_order, unique_states


def get_transition(t, u, v, transition_matrix):
    """Returns the transition probability of a tag given the previous two tags.

    Args:
        t (str): The previous previous tag.
        u (str): The previous tag.
        v (str): The tag.
        transition_matrix (dict): The transition matrix.

    Returns:
        float: The transition probability of the tag given the previous two tags.
    """

    return transition_matrix.get((t, u, v), 0)


def viterbi(sequence, transitions, states, emissions, emission_words, emission_states):
    """Viterbi algorithm for second order HMMs.

    Args:
        sequence (list): The observation sequence to decode
        transitions (dict): Transition matrix of the form {(t, u, v): count}
        states (set): The set of unique states
        emissions (??): Emission matrix of some sort (TBD) 
        emission_words (list): Observed words during training
        emission_states (list): Observed states during training

    Returns:
        list: The decoded sequence (tags)
    """

    N = len(sequence) + 2

    unique_state_tuples = get_unique_state_tuples(states)

    V = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        unique_state_tuples), columns=range(N)).fillna(0)
    B = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        unique_state_tuples), columns=range(N))

    V.loc[('PRE_START', 'START'), 0] = 1

    # Forward pass
    tweet = []
    for i in range(1, N-1):
        word = sequence[i-1]
        tweet.append(word)
        if word not in emission_words:
            word = '#UNK#'
        for v in states:
            # For current, iterate through all possible previous tuples of states leading to current
            for u in states:
                for t in states:
                    transition_prob = get_transition(t, u, v, transitions)
                    emission_prob = get_emission(
                        word, v, emissions, emission_words, emission_states)
                    # Best of previous two, leading to current
                    prob = V.loc[(t, u), i-1] * transition_prob * emission_prob
                    # Is this tuple the best so far?
                    if prob > V.loc[(u, v), i]:
                        # Update if it is
                        V.loc[(u, v), i] = prob
                        # Store grandparent to (u, v)
                        B.loc[(u, v), i] = t

    # Fill stop without emission
    v = 'STOP'
    for u in states:
        for t in states:
            transition_prob = get_transition(t, u, v, transitions)
            prob = V.loc[(t, u), N-2] * transition_prob
            if prob > V.loc[(u, v), N-1]:
                V.loc[(u, v), N-1] = prob
                B.loc[(u, v), N-1] = t

    # Backtracking
    opt_path = []
    # Determine which state it ends in (u, 'STOP'), where u is the parent of currently considered node
    curr_state = V[N-1].idxmax()
    for i in range(N-1, 0, -1):
        # Find previous (previous to parent=grandparent) state (t, u) from the grandparent matrix
        next_state = B.loc[curr_state, i]
        # Check for invalid path (outside of entities)
        if pd.isnull(next_state):
            # Tag all remaining words as outside of entity
            opt_path.extend(['O'] * i)
            break
        # Add the current state to the optimal path
        opt_path.append(curr_state[1])
        # Move backwards setting current state to (grandparent, parent)
        curr_state = (next_state, curr_state[0])

    # Reverse the path
    opt_path = opt_path[::-1][:-1]

    return opt_path


def train(lang_path):
    train_path = f"./{lang_path}/train"

    emissions, _, _, observed_values, hidden_states = generate_emission_matrix(
        f"{lang_path}/train")
    transitions_second_order, unique_states = fit_transitions_second_order(
        train_path)

    return transitions_second_order, unique_states, emissions, observed_values, hidden_states


def test(lang_path, transitions, states, emissions, emission_words, emission_states):
    test_path = f"./{lang_path}/dev.in"
    output_path = f"./{lang_path}/dev.p3.out"

    with open(test_path, 'r') as f:
        with open(output_path, 'w') as out:
            tweet = []
            lines = f.readlines()
            for line in tqdm(lines):
                # Check for end of tweet
                if line == '\n':
                    opt_path = viterbi(
                        tweet, transitions, states, emissions, emission_words, emission_states)
                    for i in range(len(tweet)):
                        out.write(f"{tweet[i]} {opt_path[i]}")
                        out.write('\n')
                    out.write('\n')
                    tweet = []
                else:
                    tweet.append(line.strip())


if __name__ == '__main__':
    try:
        lang_path = sys.argv[1]
    except:
        print("Please provide a language path as an argument (python part3.py <lang_path>). Possible values are 'EN' and 'FR' (without quotes)")

    transitions, states, emissions, emission_words, emission_states = train(
        lang_path)
    test(lang_path, transitions, states, emissions,
         emission_words, emission_states)
