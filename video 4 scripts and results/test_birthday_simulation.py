import unittest
import random
from birthday_simulation import has_shared_birthday, simulate, theoretical_probability


class BirthdayTests(unittest.TestCase):
    # --- Video 3: Classic 2-Person Tests ---
    def test_matching(self):
        self.assertFalse(has_shared_birthday([]))
        self.assertFalse(has_shared_birthday([1, 2, 365]))
        self.assertTrue(has_shared_birthday([1, 365, 1]))
        self.assertTrue(has_shared_birthday([42, 42, 42]))

    def test_boundaries(self):
        for n in [0, 1]:
            self.assertEqual(simulate(n, 20), 0)
            self.assertEqual(theoretical_probability(n), 0)
        self.assertEqual(simulate(366, 20), 1)
        self.assertEqual(theoretical_probability(366), 1)

    def test_exact(self):
        self.assertAlmostEqual(theoretical_probability(2), 1/365)
        self.assertAlmostEqual(theoretical_probability(23), .5072972343)
        self.assertLess(theoretical_probability(22), .5)

    def test_simulation_agreement(self):
        for n in [10, 23, 30]:
            self.assertLess(abs(simulate(n, 10000, random.Random(217)) - theoretical_probability(n)), .02)

    def test_reproducibility(self):
        self.assertEqual(simulate(23, 100, random.Random(217)), simulate(23, 100, random.Random(217)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            simulate(-1)
        with self.assertRaises(ValueError):
            simulate(23, 0)
        with self.assertRaises(ValueError):
            simulate(23, 10, min_matches=1)
        with self.assertRaises(ValueError):
            theoretical_probability(-1)

    # --- Video 4: Extended 3-Person Tests ---
    def test_three_way_matching(self):
        self.assertFalse(has_shared_birthday([], min_matches=3))
        # Pairs alone must not trigger a 3-way match
        self.assertFalse(has_shared_birthday([10, 10, 20, 20, 365], min_matches=3))
        # Triplets and quadruplets must trigger a 3-way match
        self.assertTrue(has_shared_birthday([10, 20, 10, 30, 10], min_matches=3))
        self.assertTrue(has_shared_birthday([42, 42, 42, 42], min_matches=3))

    def test_three_way_boundaries(self):
        # Groups smaller than 3 can never produce a 3-person match
        for n in [0, 1, 2]:
            self.assertEqual(simulate(n, 20, min_matches=3), 0)
        # Pigeonhole principle: 2 * 365 + 1 = 731 guarantees at least one birthday has 3+ people
        self.assertEqual(simulate(731, 10, min_matches=3), 1)

    def test_three_way_simulation_benchmarks(self):
        # At n=23, 3-way probability is ~1.4% (much lower than the 50.7% 2-way probability)
        p23 = simulate(23, 10000, random.Random(217), min_matches=3)
        self.assertAlmostEqual(p23, 0.014, delta=0.01)
        # At n=88, 3-way probability crosses the 50% threshold (~51.1%)
        p88 = simulate(88, 10000, random.Random(217), min_matches=3)
        self.assertAlmostEqual(p88, 0.511, delta=0.02)

    def test_three_way_reproducibility(self):
        self.assertEqual(
            simulate(88, 100, random.Random(217), min_matches=3),
            simulate(88, 100, random.Random(217), min_matches=3),
        )


if __name__ == '__main__':
    unittest.main()