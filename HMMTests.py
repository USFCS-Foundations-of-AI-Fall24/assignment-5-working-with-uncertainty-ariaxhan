# Tests for HMM

import unittest
from HMM import HMM, Sequence


class HMMTests(unittest.TestCase):

    def test_load_cat(self):
        # Test model file base name
        model_name = 'cat'
        # Create an HMM object
        hmm = HMM()
        # Load the 'cat' model
        hmm.load(model_name)
        # Test the transitions dictionary
        expected_transitions = {
            '#': {'happy': 0.5, 'grumpy': 0.5, 'hungry': 0.0},
            'happy': {'happy': 0.5, 'grumpy': 0.1, 'hungry': 0.4},
            'grumpy': {'happy': 0.6, 'grumpy': 0.3, 'hungry': 0.1},
            'hungry': {'happy': 0.1, 'grumpy': 0.6, 'hungry': 0.3}
        }

        # Check each state and its transitions
        for state, transitions in expected_transitions.items():
            for target_state, expected_value in transitions.items():
                actual_value = hmm.transitions[state][target_state]
                self.assertAlmostEqual(
                    actual_value, expected_value, places=6,
                    msg=f"Mismatch in transition {state} -> {target_state}: {actual_value} != {expected_value}"
                )

        # Test the emissions dictionary
        expected_emissions = {
            'happy': {'silent': 0.2, 'meow': 0.3, 'purr': 0.5},
            'grumpy': {'silent': 0.5, 'meow': 0.4, 'purr': 0.1},
            'hungry': {'silent': 0.2, 'meow': 0.6, 'purr': 0.2}
        }

        # Check each state and its emissions
        for state, emissions in expected_emissions.items():
            for observation, expected_value in emissions.items():
                actual_value = hmm.emissions[state][observation]
                self.assertAlmostEqual(
                    actual_value, expected_value, places=6,
                    msg=f"Mismatch in emission {state} -> {observation}: {actual_value} != {expected_value}"
                )

    # Test the generate
    def test_generate(self):
        """Test generate function for 'cat' and 'partofspeech' models."""

        # Test for 'cat' model
        model_name = 'cat'
        hmm = HMM()
        hmm.load(model_name)
        seq = hmm.generate(10)

        # Verify sequence length
        self.assertEqual(len(seq.outputseq), 10, "Generated sequence length should match the requested value.")

        # Check that all states in the sequence exist in the model's transitions
        for state in seq.stateseq:
            self.assertIn(state, hmm.transitions, f"State {state} is not valid according to the model's transitions.")

        # Check that all emissions in the sequence exist in the model's emissions
        for state, emission in zip(seq.stateseq, seq.outputseq):
            self.assertIn(emission, hmm.emissions.get(state, {}),
                          f"Emission {emission} is not valid for state {state} in the model's emissions.")

        # Test for 'partofspeech' model
        model_name = 'partofspeech'
        hmm = HMM()
        hmm.load(model_name)
        seq = hmm.generate(10)

        # Verify sequence length
        self.assertEqual(len(seq.outputseq), 10, "Generated sequence length should match the requested value.")

        # Check that all states in the sequence exist in the model's transitions
        for state in seq.stateseq:
            self.assertIn(state, hmm.transitions, f"State {state} is not valid according to the model's transitions.")

        # Check that all emissions in the sequence exist in the model's emissions
        for state, emission in zip(seq.stateseq, seq.outputseq):
            self.assertIn(emission, hmm.emissions.get(state, {}),
                          f"Emission {emission} is not valid for state {state} in the model's emissions.")

 # Test forward algorithm
    def test_forward(self):
        model_name = 'cat'
        hmm = HMM()
        hmm.load(model_name)

        # Create a sample observation sequence
        obs_seq = ['meow', 'silent', 'purr']
        seq = Sequence(stateseq=[], outputseq=obs_seq)

        # Run the forward algorithm
        forward_probs = hmm.forward(seq)

        # Ensure probabilities are calculated for all states
        self.assertTrue(all(state in forward_probs for state in hmm.transitions if state != '#'))
        self.assertTrue(all(prob >= 0 for prob in forward_probs.values()))

    # Test Viterbi algorithm
    def test_viterbi(self):
        model_name = 'cat'
        hmm = HMM()
        hmm.load(model_name)

        # Create a sample observation sequence
        obs_seq = ['meow', 'silent', 'purr']
        seq = Sequence(stateseq=[], outputseq=obs_seq)

        # Run the Viterbi algorithm
        viterbi_path = hmm.viterbi(seq)

        # Ensure a valid sequence of states is returned
        self.assertEqual(
            len(viterbi_path), len(obs_seq),
            msg=f"Viterbi path length mismatch: {len(viterbi_path)} != {len(obs_seq)}. Path: {viterbi_path}"
        )
        self.assertTrue(
            all(state in hmm.transitions for state in viterbi_path),
            msg=f"Invalid state in Viterbi path: {viterbi_path}"
        )


    def test_transition_probabilities_sum(self):
        """Test that transition probabilities for each state sum to 1."""
        hmm = HMM()
        hmm.load("cat")
        for state, transitions in hmm.transitions.items():
            total_prob = sum(transitions.values())
            self.assertAlmostEqual(total_prob, 1.0, places=6, msg=f"Transition probabilities for {state} do not sum to 1")

    def test_emission_probabilities_sum(self):
        """Test that emission probabilities for each state sum to 1."""
        hmm = HMM()
        hmm.load("cat")
        for state, emissions in hmm.emissions.items():
            total_prob = sum(emissions.values())
            self.assertAlmostEqual(total_prob, 1.0, places=6, msg=f"Emission probabilities for {state} do not sum to 1")

    def test_main_integration(self):
        """Test integration of generate, forward, and viterbi commands."""
        hmm = HMM()
        hmm.load("cat")
        seq = hmm.generate(10)
        obs_seq = seq.outputseq

        # Run forward
        forward_probs = hmm.forward(Sequence(stateseq=[], outputseq=obs_seq))
        self.assertTrue(all(prob >= 0 for prob in forward_probs.values()), "Forward probabilities should be non-negative")

        # Run Viterbi
        viterbi_path = hmm.viterbi(Sequence(stateseq=[], outputseq=obs_seq))
        self.assertEqual(len(viterbi_path), len(obs_seq), "Viterbi path length should match observation sequence length")

    def test_lander(self):
        """Test the full pipeline for the 'lander' model, including load, generate, forward, and viterbi."""

        # Load the 'lander' model
        model_name = 'lander'
        hmm = HMM()
        hmm.load(model_name)
        # Generate a sequence
        sequence_length = 10
        seq = hmm.generate(sequence_length)

        # Validate generated sequence
        self.assertEqual(len(seq.outputseq), sequence_length,
                         "Generated sequence length should match the requested value.")
        for state in seq.stateseq:
            self.assertIn(state, hmm.transitions, f"State {state} is not valid according to the model's transitions.")
        for state, emission in zip(seq.stateseq, seq.outputseq):
            self.assertIn(emission, hmm.emissions.get(state, {}),
                          f"Emission {emission} is not valid for state {state} in the model's emissions.")

        # Run forward algorithm
        forward_probs = hmm.forward(seq)
        self.assertTrue(all(state in forward_probs for state in hmm.transitions if state != '#'),
                        "Forward probabilities should be calculated for all states.")
        self.assertTrue(all(prob >= 0 for prob in forward_probs.values()),
                        "Forward probabilities should be non-negative.")

        # Run Viterbi algorithm
        viterbi_path = hmm.viterbi(seq)
        self.assertEqual(
            len(viterbi_path), len(seq.outputseq),
            msg=f"Viterbi path length mismatch: {len(viterbi_path)} != {len(seq.outputseq)}"
        )
        self.assertTrue(
            all(state in hmm.transitions for state in viterbi_path),
            msg=f"Invalid state in Viterbi path: {viterbi_path}"
        )

if __name__ == '__main__':
    unittest.main()
