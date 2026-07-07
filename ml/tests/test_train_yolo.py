"""
Unit tests for the YOLO training helpers:
    python -m pytest ml/tests -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_yolo import iou_yolo, largest_bbox, write_dataset_yaml  # noqa: E402


class TestIouYolo:
    def test_identical_boxes(self):
        b = np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)
        assert iou_yolo(b, b)[0] == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        a = np.array([[0.2, 0.2, 0.1, 0.1]], dtype=np.float32)
        b = np.array([[0.8, 0.8, 0.1, 0.1]], dtype=np.float32)
        assert iou_yolo(a, b)[0] == 0.0

    def test_half_overlap(self):
        # box b = right half of a -> IoU 0.5
        a = np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)
        b = np.array([[0.6, 0.5, 0.2, 0.2]], dtype=np.float32)
        assert iou_yolo(a, b)[0] == pytest.approx(0.5, abs=1e-6)

    def test_vectorized(self):
        a = np.tile([0.5, 0.5, 0.2, 0.2], (3, 1)).astype(np.float32)
        assert iou_yolo(a, a).shape == (3,)


class TestLargestBbox:
    def _dataset(self, tmp_path, label_lines):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        images.mkdir(), labels.mkdir()
        img = images / "car.jpg"
        img.write_bytes(b"fake")
        (labels / "car.txt").write_text(label_lines, encoding="utf-8")
        return str(img)

    def test_picks_largest_area(self, tmp_path):
        img = self._dataset(tmp_path,
                            "0 0.3 0.3 0.10 0.10\n"
                            "0 0.6 0.6 0.40 0.30\n"   # largest area
                            "0 0.5 0.5 0.20 0.20\n")
        assert largest_bbox(img) == (0.6, 0.6, 0.40, 0.30)

    def test_no_label_file(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        img = images / "lonely.jpg"
        img.write_bytes(b"fake")
        assert largest_bbox(str(img)) is None

    def test_malformed_lines_ignored(self, tmp_path):
        img = self._dataset(tmp_path, "garbage\n0 0.5 0.5 0.2 0.1\n")
        assert largest_bbox(img) == (0.5, 0.5, 0.2, 0.1)


class TestDatasetYaml:
    def test_valid_alias(self, tmp_path):
        (tmp_path / "valid").mkdir()
        path = write_dataset_yaml(str(tmp_path))
        content = open(path, encoding="utf-8").read()
        assert "val: valid/images" in content
        assert "license-plate" in content
