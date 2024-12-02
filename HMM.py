

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
            current_state = None
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                # Handle the starting state marker (#)
                if line.startswith('#'):
                    current_state = '#'
                    self.transitions[current_state] = {}
                    self.transitions[current_state][parts[1]] = float(parts[2])
                    continue

                # Transition lines
                from_state, to_state, prob = parts
                if from_state not in self.transitions:
                    self.transitions[from_state] = {}
                self.transitions[from_state][to_state] = float(prob)

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
        # repeatedly select successor states at random
        for i in range(n):
            # select the next state using random choice, with the transition probability as a weight
            state = numpy.random.choice(list(self.transitions[state].keys()), p=[float(x) for x in self.transitions[state].values()])
            # select an emission using random choice, with the emission probability as a weight
            output = numpy.random.choice(list(self.emissions[state].keys()), p=[float(x) for x in self.emissions[state].values()])
            # append the state and output to the sequences
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
        for state in self.transitions:
            M[state][0] = float(self.transitions['#'][state]) * float(self.emissions[state][sequence.outputseq[0]])

        # Iterate over each subsequent item in the sequence starting from the second item (i=1)
        for i in range(1, len(sequence.outputseq)):
            for s in self.transitions:
                # Initialize the sum for the forward probability of state `s` at item `i`
                sum_prob = 0
                for s_prev in self.transitions:
                    # Compute the contribution of the previous state `s_prev` to `s` at item `i`
                    transition_prob = float(self.transitions[s_prev][s])
                    emission_prob = float(self.emissions[s][sequence.outputseq[i]])
                    sum_prob += M[s_prev][i - 1] * transition_prob * emission_prob

                # Store the computed forward probability in `M[s][i]`
                M[s][i] = sum_prob

        # Return the forward probabilities at the final item for each state
        return {state: M[state][-1] for state in self.transitions}

    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.


    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.
    #




# Main function
def main():
    # Parse command line arguments
    # example: python hmm.py cat --generate 20 --forward generated.obs
    parser = argparse.ArgumentParser(description='HMM')
    parser.add_argument('model', help='Model name')
    parser.add_argument('--generate', type=int, help='Generate a sequence of length n')
    parser.add_argument('--forward', type=str, help='Run forward algorithm on an observation sequence')
    args = parser.parse_args()

    # Initialize and load the HMM
    hmm = HMM()
    print("Loading model from files: ", args.model)
    hmm.load(args.model)
    print("Transitions: ", hmm.transitions)
    print("Emissions: ", hmm.emissions)

    # Generate a sequence if requested
    if args.generate:
        print("Generating a sequence of length ", args.generate)
        seq = hmm.generate(args.generate)
        print("Generated sequence:")
        print(seq)

        # Save the generated observation sequence
        obs_filename = f"{args.model}_sequence.obs"
        hmm.save_obs(seq, obs_filename)
        print(f"Observation sequence saved to {obs_filename}")

    # Run the forward algorithm if requested
    if args.forward:
        # Load the observation sequence
        print(f"Running forward algorithm on observation sequence from {args.forward}")
        with open(args.forward, 'r') as f:
            obs_seq = [line.strip() for line in f.readlines()]

        # Create a Sequence object with placeholder states (states are unknown here)
        seq = Sequence(stateseq=[], outputseq=obs_seq)

        # Run the forward algorithm
        forward_probs = hmm.forward(seq)
        print("Forward probabilities:")
        for state, prob in forward_probs.items():
            print(f"State {state}: {prob:.6f}")

if __name__ == '__main__':
    main()
