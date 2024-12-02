

import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)



class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        
        # Reset dictionaries
        self.transitions = {}
        self.emissions = {}
        # Load transitions
        with open(basename + '.trans', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                # Handle the starting state marker (#)
                if line.startswith('#'):
                    current_state = '#'
                    if current_state not in self.transitions:
                        self.transitions[current_state] = {}
                    self.transitions[current_state][parts[1]] = float(parts[2])
                    continue

                # Transition lines
                from_state, to_state, prob = parts
                if from_state not in self.transitions:
                    self.transitions[from_state] = {}
                self.transitions[from_state][to_state] = float(prob)

        # Normalize transition probabilities
        for state, transitions in self.transitions.items():
            total = sum(transitions.values())
            if total > 0:
                self.transitions[state] = {k: v / total for k, v in transitions.items()}

        # Load emissions
        with open(basename + '.emit', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    state, output, prob = parts
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][output] = float(prob)




   ## you do this.
    ## take an integer n, and return a Sequence of length n.
    # To generate this, start in the initial state and repeatedly select successor states at random,
    # using the transition probability as a weight, and then select an emission,
    # using the emission probability as a weight. You may find either numpy.random.choice or
    # random.choices very helpful here. Be sure that you are using the transition probabilities
    # to determine the next state, and not a uniform distribution
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        # start in the initial state
        state = '#'
        stateseq = []
        outputseq = []

        # Check transitions from '#'
        if state in self.transitions and sum(self.transitions[state].values()) > 0:
            transitions = self.transitions[state]
            probabilities = [prob / sum(transitions.values()) for prob in transitions.values()]
            state = numpy.random.choice(list(transitions.keys()), p=probabilities)
        else:
            raise ValueError(f"Initial state '{state}' has no valid transitions or is undefined.")

        # Generate the sequence
        for _ in range(n):
            # Handle state transitions
            transitions = self.transitions.get(state, {})
            total = sum(transitions.values())
            if total > 0:
                probabilities = [prob / total for prob in transitions.values()]
                state = numpy.random.choice(list(transitions.keys()), p=probabilities)
            else:
                raise ValueError(f"State '{state}' has no valid transitions.")

            # Handle emissions
            emissions = self.emissions.get(state, {})
            total = sum(emissions.values())
            if total > 0:
                probabilities = [prob / total for prob in emissions.values()]
                output = numpy.random.choice(list(emissions.keys()), p=probabilities)
            else:
                raise ValueError(f"State '{state}' has no valid emissions.")

            stateseq.append(state)
            outputseq.append(output)

        # return the sequence
        return Sequence(stateseq, outputseq)

    # Function to save a Sequence to a file in a proper obs format
    def save_obs(self, seq, filename):
        """
        Saves the output sequence of a given Sequence object to a file.

        Args:
            seq (Sequence): The sequence to save, containing the output sequence.
            filename (str): The name of the file to save the sequence to.
        """
        try:
            # Open the file in write mode
            with open(filename, 'w') as f:
                # Iterate through the output sequence and write each output to a new line
                for output in seq.outputseq:
                    f.write(output + '\n')
            print(f"Sequence successfully saved to {filename}.")
        except Exception as e:
            print(f"Error saving sequence to {filename}: {e}")

    # This tells us, for a sequence of observations, the most likely final state.
    # Forward should predict the most probable state given the sequence of emisssions.
    # For the lander, please indicate whether it's safe to land or not.
    def forward(self, sequence):
        # Initialize the forward probability matrix `M`
        # `M[state][i]` will store the probability of being in `state` at item `i` in the sequence
        M = {state: [0] * len(sequence.outputseq) for state in self.transitions}

        # Set initial probabilities for the first item in the sequence (i=0), considering the '#' start state
        for state in self.transitions.get('#', {}):
            if sequence.outputseq[0] in self.emissions.get(state, {}):
                initial_prob = self.transitions['#'][state] * self.emissions[state][sequence.outputseq[0]]
                M[state][0] = initial_prob
                print(f"Initial probability for state {state}: {initial_prob}")

        # Iterate over each subsequent item in the sequence starting from the second item (i=1)
        for i in range(1, len(sequence.outputseq)):
            for s in [state for state in self.transitions if state != '#']:
                # Initialize the sum for the forward probability of state `s` at item `i`
                sum_prob = 0
                for s_prev in self.transitions:
                    # Compute the contribution of the previous state `s_prev` to `s` at item `i`
                    if s in self.transitions.get(s_prev, {}) and sequence.outputseq[i] in self.emissions.get(s, {}):
                        transition_prob = float(self.transitions[s_prev][s])
                        emission_prob = float(self.emissions[s][sequence.outputseq[i]])
                        contrib = M[s_prev][i - 1] * transition_prob * emission_prob
                        sum_prob += contrib
                        print(f"Contribution to state {s} at time {i} from state {s_prev}: {contrib:.6f}")

                # Store the computed forward probability in `M[s][i]`
                M[s][i] = sum_prob
                print(f"Forward probability for state {s} at time {i}: {sum_prob:.6f}")

        # Return the forward probabilities at the final item for each state
        return {state: M[state][-1] for state in self.transitions}

    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.


    def viterbi(self, sequence):
        """
        Viterbi algorithm to find the most likely sequence of states
        given an observation sequence.
        """
        # Initialize Viterbi table and path trackers
        V = {state: [0] * len(sequence.outputseq) for state in self.transitions if state != '#'}
        path = {state: [] for state in V}

        # Initialize base cases (t=0)
        for state in self.transitions.get('#', {}):
            if sequence.outputseq[0] in self.emissions.get(state, {}):
                V[state][0] = (
                        self.transitions['#'][state] * self.emissions[state][sequence.outputseq[0]]
                )
                path[state] = [state]

        # Fill in Viterbi table
        for t in range(1, len(sequence.outputseq)):
            for s in V:
                max_prob, best_prev = max(
                    (V[prev_state][t - 1] * self.transitions[prev_state].get(s, 0) *
                     self.emissions[s].get(sequence.outputseq[t], 0), prev_state)
                    for prev_state in V
                )
                V[s][t] = max_prob
                path[s] = path[best_prev] + [s]

        # Find the best final state
        max_final_prob, best_final_state = max((V[state][-1], state) for state in V)

        return path[best_final_state], max_final_prob




# Main function
def main():
    # Parse command line arguments
    # example: python hmm.py cat --generate 20 --forward generated.obs
    parser = argparse.ArgumentParser(description='HMM')
    parser.add_argument('model', help='Model name')
    parser.add_argument('--generate', type=int, help='Generate a sequence of length n')
    parser.add_argument('--forward', type=str, help='Run forward algorithm on an observation sequence')
    parser.add_argument('--viterbi', type=str, help='Run Viterbi algorithm on an observation sequence')
    args = parser.parse_args()

    # Initialize and load the HMM
    hmm = HMM()
    print("Loading model from files: ", args.model)
    hmm.load(args.model)
    print("Transitions: ", hmm.transitions)
    print("Emissions: ", hmm.emissions)

    # Generate a sequence if requested
    if args.generate:
        print(f"Transitions from '#': {hmm.transitions.get('#')}")
        print("Generating a sequence of length ", args.generate)
        seq = hmm.generate(args.generate)
        print("Generated sequence:")
        print(seq)

    if args.forward:
        # Generate a sequence for the specified model and save it to the specified file
        generated_file = args.forward
        print(f"Generating a sequence for {args.model} and saving it to {generated_file}...")

        seq = hmm.generate(20)  # Adjust the length of the sequence as needed
        hmm.save_obs(seq, generated_file)  # Use the save_obs method here

        # Load the observation sequence from the generated file
        print(f"Running forward algorithm on observation sequence from {generated_file}...")
        with open(generated_file, 'r') as f:
            obs_seq = [line.strip() for line in f.readlines()]

        # Create a Sequence object with placeholder states (states are unknown here)
        seq = Sequence(stateseq=[], outputseq=obs_seq)

        # Run the forward algorithm
        forward_probs = hmm.forward(seq)
        print("Forward probabilities:")
        for state, prob in forward_probs.items():
            print(f"State {state}: {prob:.6f}")

    # Run the Viterbi algorithm if requested
    if args.viterbi:
        print(f"Running Viterbi algorithm on observation sequence from {args.viterbi}")
        with open(args.viterbi, 'r') as f:
            obs_seq = [line.strip() for line in f.readlines()]

        # Create a Sequence object with placeholder states (states are unknown here)
        seq = Sequence(stateseq=[], outputseq=obs_seq)

        # Run the Viterbi algorithm
        most_likely_path, max_prob = hmm.viterbi(seq)
        print("Most likely sequence of states:")
        print(' '.join(most_likely_path))
        print(f"Probability of the sequence: {max_prob:.6f}")

if __name__ == '__main__':
    main()
