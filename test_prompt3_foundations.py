import unittest

from Sliding_Window_Algebraic_Profiler import (
    DEFAULT_PROJECTION_VARIABLES,
    SymPyGroebnerBackend,
    TraceAlgebra,
    Word,
    build_presentation,
    check_aperiodicity,
    compute_ideal_polynomials,
    compute_character_variety,
    compute_trace_polynomials,
    generate_seed_centered_windows,
)


class FoundationTests(unittest.TestCase):
    def test_fibonacci_rule_is_aperiodic(self) -> None:
        result = check_aperiodicity("AUAU", {"A": "AU", "U": "A"})
        self.assertEqual(result["rule_source"], "user")
        self.assertTrue(result["is_primitive"])
        self.assertTrue(result["pf_is_irrational"])
        self.assertTrue(result["is_aperiodic"])

    def test_rational_pf_rule_is_not_aperiodic(self) -> None:
        result = check_aperiodicity("AUAU", {"A": "AU", "U": "AU"})
        self.assertTrue(result["is_primitive"])
        self.assertFalse(result["pf_is_irrational"])
        self.assertFalse(result["is_aperiodic"])

    def test_heuristic_rule_is_reported(self) -> None:
        result = check_aperiodicity("UCCUAC")
        self.assertEqual(result["rule_source"], "heuristic-cyclic-successor")
        self.assertIn("U", result["rule"])

    def test_trace_algebra_is_cyclically_invariant(self) -> None:
        presentation = build_presentation("UCCUAC")
        algebra = TraceAlgebra(presentation)
        self.assertEqual(algebra.reduce("UCCUAC"), algebra.reduce("CCUACU"))

    def test_trace_algebra_is_inverse_invariant(self) -> None:
        presentation = build_presentation("UCCUAC")
        algebra = TraceAlgebra(presentation)
        inverse_word = Word.from_sequence("UCCUAC").inverse()
        self.assertEqual(algebra.reduce("UCCUAC"), algebra.reduce_word(inverse_word.render()))

    def test_generate_seed_centered_windows_overlap_canonical_seed(self) -> None:
        windows = generate_seed_centered_windows("UAAGGCACGCGG", (6, 7, 8))
        self.assertTrue(windows)
        self.assertEqual(sorted(set(window["start"] for window in windows)), [0, 1, 2])

    def test_trace_polynomials_include_reduced_relator_trace(self) -> None:
        trace_data = compute_trace_polynomials("AAGGCA")
        self.assertIn("reduced_relator_trace", trace_data)
        self.assertTrue(trace_data["reduced_relator_trace"])
        self.assertIn("tr_A", trace_data["trace_coordinate_map"])

    def test_length_three_words_use_canonical_tiebreaker(self) -> None:
        presentation = build_presentation("AGCAGCGG")
        algebra = TraceAlgebra(presentation)
        reduced = algebra.reduce("AGC")
        self.assertTrue(str(reduced).startswith("tr_") or reduced.free_symbols)

    def test_agcagcgg_uses_paper_generator_order(self) -> None:
        presentation = build_presentation("AGCAGCGG")
        self.assertEqual(presentation.generators, ("A", "C", "G"))

    def test_ideal_polynomials_follow_relator_rule(self) -> None:
        presentation = build_presentation("UCCUAC")
        algebra = TraceAlgebra(presentation)
        ideal = compute_ideal_polynomials(presentation, algebra)
        self.assertEqual(len(ideal), 1 + len(presentation.generators))
        self.assertTrue(all(polynomial != 0 for polynomial in ideal))

    def test_sympy_backend_runs_on_generated_ideal(self) -> None:
        x, y = DEFAULT_PROJECTION_VARIABLES[:2]
        basis = SymPyGroebnerBackend().compute_basis([x**2 + y, x - 1], (x, y))
        self.assertIsInstance(basis, list)

    def test_character_variety_reports_backend_and_ideal(self) -> None:
        result = compute_character_variety("GGAGUGU")
        self.assertIn(result["backend"], {"sympy", "singular-subprocess"})
        self.assertIn("ideal_polynomials", result)
        self.assertEqual(result["analysis_status"], "benchmark-sliced")

    def test_active_trace_variables_expand_beyond_projection_ring(self) -> None:
        result = compute_character_variety("GGAGUGU")
        active = set(result["active_trace_variables"])
        self.assertTrue(set(str(symbol) for symbol in DEFAULT_PROJECTION_VARIABLES).issubset(active))
        self.assertGreater(len(active), len(DEFAULT_PROJECTION_VARIABLES))

    def test_computed_ideal_no_longer_collapses_for_ggagugu(self) -> None:
        result = compute_character_variety("GGAGUGU")
        self.assertNotEqual(result["basis_equations"], ["1"])

    def test_agcagcgg_acceptance_case_has_nontrivial_basis(self) -> None:
        result = compute_character_variety("AGCAGCGG")
        self.assertNotEqual(result["basis_equations"], ["1"])
        self.assertEqual(result["analysis_status"], "benchmark-sliced")

    def test_agcagcgg_reports_literature_coordinate_map(self) -> None:
        result = compute_character_variety("AGCAGCGG")
        self.assertEqual(result["variable_map"]["presentation_generators"], ["A", "C", "G"])
        self.assertEqual(result["variable_map"]["literature_trace_generators"]["alpha"], "tr_A")
        self.assertEqual(result["variable_map"]["literature_trace_generators"]["beta"], "tr_C")
        self.assertEqual(result["variable_map"]["literature_trace_generators"]["gamma"], "tr_G")
        self.assertEqual(result["variable_map"]["projection_trace_generators"], {"x": "tr_AC", "y": "tr_CG", "z": "tr_AG"})
        self.assertEqual(result["variable_map"]["projection_coordinate_symbols"], {"x": "x", "y": "y", "z": "z"})
        self.assertEqual(result["intended_projection_ring"], ["x", "y", "z"])

    def test_projection_variables_are_not_listed_as_eliminated(self) -> None:
        result = compute_character_variety("AGCAGCGG")
        for sliced in result["sliced_bases"]:
            self.assertEqual(sliced["elimination_order"], ["tr_ACG"])
            self.assertEqual(sliced["groebner_variable_order"], ["tr_ACG", "x", "y", "z"])

    def test_agcagcgg_extracts_cayley_component_on_zero_slice(self) -> None:
        result = compute_character_variety("AGCAGCGG")
        zero_slice = next(s for s in result["sliced_bases"] if s["slice"] == "(0,0,0,0)")
        self.assertEqual(zero_slice["common_surface_factors"], ["x**2 + x*y*z + y**2 + z**2 - 4"])
        self.assertEqual(zero_slice["normalized_core_surface_candidates"], ["x**2 + x*y*z + y**2 + z**2 - 4"])
        self.assertTrue(zero_slice["benchmark_comparison"]["matched"])
        self.assertTrue(zero_slice["canonical_fricke_comparison"]["matched"])
        self.assertEqual(zero_slice["slice_audit_status"], "Recovered canonical slice and draft claim")

    def test_agcagcgg_1100_slice_extracts_wrong_cubic_for_kappa3_claim(self) -> None:
        result = compute_character_variety("AGCAGCGG")
        target_slice = next(s for s in result["sliced_bases"] if s["slice"] == "(1,1,0,0)")
        self.assertEqual(target_slice["common_surface_factors"], ["x**2 + x*y*z - x + y**2 + z**2 - 2"])
        self.assertTrue(target_slice["canonical_fricke_comparison"]["matched"])
        self.assertFalse(target_slice["benchmark_comparison"]["matched"])
        self.assertEqual(target_slice["slice_audit_status"], "Recovered canonical slice")

    def test_agcagcgg_1111_slice_has_no_common_cubic_component(self) -> None:
        result = compute_character_variety("AGCAGCGG")
        target_slice = next(s for s in result["sliced_bases"] if s["slice"] == "(1,1,1,1)")
        self.assertEqual(target_slice["common_surface_factors"], [])
        self.assertFalse(target_slice["benchmark_comparison"]["matched"])
        self.assertFalse(target_slice["canonical_fricke_comparison"]["matched"])
        self.assertEqual(target_slice["slice_audit_status"], "Only lower-dimensional components present")

    def test_agcagcgg_is_tagged_as_legacy_benchmark(self) -> None:
        result = compute_character_variety("AGCAGCGG")
        self.assertEqual(result["benchmark_status"], "legacy-draft-audit")
        self.assertEqual(result["published_benchmark_recommendations"], ["UCCUAC", "UCCUACA", "AAGGCA", "AAGGCAC"])

    def test_healthy_control_allows_curves_only(self) -> None:
        result = compute_character_variety("GGAGUGU")
        self.assertEqual(result["singularity_type"], "None")
        self.assertEqual(result["xyz_surface_factors"], [])


if __name__ == "__main__":
    unittest.main()
