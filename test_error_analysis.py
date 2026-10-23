"""Checks for error categories and aggregate entity counts."""

import unittest

from analyze_dev_errors import classify_miss, length_bucket, summarize


class ErrorAnalysisTests(unittest.TestCase):
    def test_miss_categories(self):
        gold = ("ORG", 2, 6)
        self.assertEqual(classify_miss(gold, {("PER", 2, 6)}), "type_error")
        self.assertEqual(classify_miss(gold, {("ORG", 3, 6)}), "boundary_error")
        self.assertEqual(classify_miss(gold, {("LOC", 3, 6)}), "boundary_and_type")
        self.assertEqual(classify_miss(gold, {("LOC", 6, 8)}), "not_detected")

    def test_length_buckets(self):
        self.assertEqual([length_bucket(x) for x in (1, 2, 3, 4, 5, 8, 9)],
                         ["1-2", "1-2", "3-4", "3-4", "5-8", "5-8", "9+"])
        with self.assertRaises(ValueError):
            length_bucket(0)

    def test_aggregate_counts_match_entity_f1(self):
        gold = {("PER", 0, 2), ("ORG", 2, 6), ("LOC", 6, 8)}
        predicted = {("PER", 0, 2), ("PER", 2, 6), ("LOC", 6, 7)}
        report = summarize([("张三北京大学上海", predicted, gold)])
        self.assertEqual(report["overall"]["gold"], 3)
        self.assertEqual(report["overall"]["predicted"], 3)
        self.assertEqual(report["overall"]["correct"], 1)
        self.assertAlmostEqual(report["overall"]["f1"], 1 / 3)
        self.assertEqual(report["missed_gold_categories"]["ORG"]["type_error"], 1)
        self.assertEqual(report["missed_gold_categories"]["LOC"]["boundary_error"], 1)
        self.assertEqual(report["by_length"]["3-4"]["gold"], 1)
        self.assertEqual(report["by_type"]["PER"]["predicted"], 2)
        self.assertAlmostEqual(report["by_type"]["PER"]["f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
