import unittest

from Sliding_Window_Algebraic_Profiler import compute_character_variety, run_regression_tests


class RegressionTests(unittest.TestCase):
    def test_published_regression_suite(self) -> None:
        results = run_regression_tests()
        self.assertTrue(all(results.values()))

    def test_uccuac_is_singular(self) -> None:
        result = compute_character_variety("UCCUAC")
        self.assertTrue(result["is_singular"])
        self.assertEqual(result["singularity_type"], "A1 f_b")

    def test_uccuaca_is_nonsingular(self) -> None:
        result = compute_character_variety("UCCUACA")
        self.assertFalse(result["is_singular"])
        self.assertEqual(result["singularity_type"], "None")

    def test_aaggca_is_singular(self) -> None:
        result = compute_character_variety("AAGGCA")
        self.assertTrue(result["is_singular"])
        self.assertEqual(result["singularity_type"], "A1 f_(2,{})")

    def test_aaggcac_is_nonsingular(self) -> None:
        result = compute_character_variety("AAGGCAC")
        self.assertFalse(result["is_singular"])
        self.assertEqual(result["singularity_type"], "None")


if __name__ == "__main__":
    unittest.main()
