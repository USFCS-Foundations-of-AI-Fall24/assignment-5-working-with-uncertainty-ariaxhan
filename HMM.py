

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

        # Read transitions
        with open(basename + '.trans', 'r') as f:
            # Read each line
            for line in f:
                # Split the line into parts
                parts = line.strip().split()
                # If there are three parts
                if len(parts) == 3:
                    # Assign the parts to variables
                    from_state, to_state, prob = parts
                    # If the from_state is not in the transitions dictionary
                    if from_state not in self.transitions:
                        # Create a new dictionary
                        self.transitions[from_state] = {}
                    # Assign the probability to the to_state
                    self.transitions[from_state][to_state] = prob

        # Read emissions
        with open(basename + '.emit', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    state, output, prob = parts
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][output] = prob


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        pass

    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.


    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.
    #




# Main function
def main():
    # Test model file base name, test with cat first
    model_name = 'cat'
    # Initialize and load the HMM
    hmm = HMM()
    print("Loading model from files: ", model_name)
    hmm.load(model_name)
    print("Transitions: ", hmm.transitions)
    print("Emissions: ", hmm.emissions)



if __name__ == '__main__':
    main()
