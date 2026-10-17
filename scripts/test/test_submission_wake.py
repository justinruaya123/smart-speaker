"""Wake selection retains true phrase identities and explicit fallback behavior."""
import unittest
import numpy as np
from unittest.mock import Mock
from scripts.submission.wake import WakeChoice


class WakeChoiceTests(unittest.TestCase):
    def test_fallback_accepts_either_actual_detector_with_correct_phrase(self):
        iris=Mock();jarvis=Mock();iris.push.return_value=0.;jarvis.push.return_value=.8
        detector=WakeChoice(iris,jarvis)
        self.assertEqual(detector.push(np.zeros(512)),.8)
        self.assertEqual(detector.last_phrase,'hey jarvis')
        iris.push.return_value=1.;jarvis.push.return_value=.1
        self.assertEqual(detector.push(np.zeros(512)),1.)
        self.assertEqual(detector.last_phrase,'hey iris')
        detector.reset();iris.reset.assert_called_once();jarvis.reset.assert_called_once()

    def test_iris_only_does_not_invoke_jarvis_and_jarvis_only_is_explicit(self):
        iris=Mock();iris.push.return_value=1.
        self.assertEqual(WakeChoice(iris,None).push(np.zeros(512)),1.)
        jarvis=Mock();jarvis.push.return_value=.6
        self.assertEqual(WakeChoice(None,jarvis).push(np.zeros(512)),.6)

    def test_iris_inference_failure_switches_to_jarvis_only_in_fallback_mode(self):
        iris=Mock();iris.push.side_effect=RuntimeError('broken graph');jarvis=Mock();jarvis.push.return_value=.7
        detector=WakeChoice(iris,jarvis)
        self.assertEqual(detector.push(np.zeros(512)),.7);self.assertIsNone(detector.iris)
        self.assertIn('Jarvis is active',detector.warning)
        with self.assertRaises(RuntimeError):WakeChoice(iris,None).push(np.zeros(512))


if __name__=='__main__':unittest.main()
