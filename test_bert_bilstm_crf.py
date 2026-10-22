"""Checks for the sequence decoder and leak-free data split."""

from itertools import product
import math
import unittest

import torch

from bert_bilstm_crf import LinearCRF, entities, sample_splits
from demo_server import boundary_windows, merge_boundary_rows, segment_text


class CrfTests(unittest.TestCase):
    def test_long_text_segments_keep_original_offsets(self):
        text = "张三在北京大学。\n李华 加入中国科学院，后来去了上海。" + "北京大学" * 30
        parts = segment_text(text, 24)
        self.assertTrue(all(0 < len(part) <= 24 for _, part in parts))
        self.assertEqual("".join(part for _, part in parts), "".join(text.split()))
        self.assertTrue(all(text[offset:offset + len(part)] == part for offset, part in parts))
        self.assertTrue(any(offset > text.index("上海") for offset, _ in parts))

    def test_boundary_context_and_replacement(self):
        text = "研究团队" * 3 + "北京大学" + "继续研究"
        parts = segment_text(text, 14)
        windows = boundary_windows(text, parts, 14)
        self.assertTrue(all(len(part) <= 14 and text[start:start + len(part)] == part
                            for _, start, part in windows))
        self.assertTrue(any(12 < boundary < 16 and start < boundary < start + len(part)
                            for boundary, start, part in windows))
        original = [{"text": "北京", "label": "地名", "start": 12, "end": 14},
                    {"text": "继续", "label": "地名", "start": 16, "end": 18}]
        recovered = [{"text": "北京大学", "label": "机构名", "start": 12, "end": 16}]
        self.assertEqual(merge_boundary_rows(original, recovered),
                         [recovered[0], original[1]])

    def test_partition_and_decode_match_brute_force(self):
        torch.manual_seed(17)
        crf = LinearCRF(3)
        with torch.no_grad():
            crf.start.copy_(torch.randn(3))
            crf.end.copy_(torch.randn(3))
            crf.transition.copy_(torch.randn(3, 3))
        emission = torch.randn(2, 3, 3, requires_grad=True)
        labels = torch.tensor([[0, 2, 1], [1, 0, 0]])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        brute_losses = []
        for row, length in enumerate([3, 2]):
            scores = {}
            for path in product(range(3), repeat=length):
                score = crf.start[path[0]] + emission[row, 0, path[0]]
                for t in range(1, length):
                    score = score + crf.transition[path[t - 1], path[t]] + emission[row, t, path[t]]
                scores[path] = score + crf.end[path[-1]]
            actual = tuple(labels[row, :length].tolist())
            brute_losses.append(torch.logsumexp(torch.stack(list(scores.values())), 0) - scores[actual])
            self.assertEqual(crf.decode(emission.detach(), mask)[row], list(max(scores, key=lambda p: float(scores[p].detach()))))
        expected = torch.stack(brute_losses).mean()
        observed = crf.loss(emission, labels, mask)
        self.assertTrue(math.isclose(float(expected.detach()), float(observed.detach()), rel_tol=1e-6))
        observed.backward()
        self.assertTrue(torch.isfinite(emission.grad).all())

    def test_entity_spans(self):
        self.assertEqual(entities("张三在北京", ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]),
                         {("PER", 0, 2), ("LOC", 3, 5)})

    def test_splits_disjoint_and_deterministic(self):
        train = [{"line": i} for i in range(100)]
        test = [{"line": i} for i in range(1000, 1050)]
        a = sample_splits(train, test, 20, 10, 8, 42)
        b = sample_splits(train, test, 20, 10, 8, 42)
        self.assertEqual(a, b)
        self.assertTrue(set(x["line"] for x in a[0]).isdisjoint(x["line"] for x in a[1]))
        self.assertTrue(all(x["line"] >= 1000 for x in a[2]))
        excluded = {x["line"] for x in a[2]}
        c = sample_splits(train, test, 20, 10, 8, 43, excluded)
        self.assertTrue(excluded.isdisjoint(x["line"] for x in c[2]))


if __name__ == "__main__":
    unittest.main()
