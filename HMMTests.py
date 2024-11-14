# Tests for HMM

import unittest
from HMM import HMM


class HMMTests(unittest.TestCase):
    # Test the load function with the 'cat' model
    def test_load_cat(self):
        # Test model file base name
        model_name = 'cat'
        # Create an HMM object
        hmm = HMM()
        # Load the 'cat' model
        hmm.load(model_name)
        # Test the transitions dictionary
        expected_transitions = {
            '#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'},
            'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'},
            'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'},
            'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}
        }
        self.assertEqual(hmm.transitions, expected_transitions)
        # Test the emissions dictionary
        expected_emissions = {
            'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
            'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
            'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}
        }
        self.assertEqual(hmm.emissions, expected_emissions)

    # Test the generate
    def test_generate(self):
        # Test model file base name
        model_name = 'cat'
        # Create an HMM object
        hmm = HMM()
        # Load the 'cat' model
        hmm.load(model_name)
        # Generate a sequence
        seq = hmm.generate(10)
        # Test the sequence
        self.assertEqual(len(seq), 10)

        # Now try it for the 'partofspeech'
        model_name = 'partofspeech'
        # Create an HMM object
        hmm = HMM()
        # Load the 'partofspeech' model
        hmm.load(model_name)
        # Generate a sequence
        seq = hmm.generate(10)
        # Test the sequence
        self.assertEqual(len(seq), 10)



if __name__ == '__main__':
    unittest.main()
