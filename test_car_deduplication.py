"""Verification tests for single most accurate car box proposal rule."""

import unittest
from vision import filter_one_car_per_image, deduplicate_car_predictions, box_iou


class TestCarSingleMostAccurateBox(unittest.TestCase):
    def test_1_highest_scoring_box_retained_even_if_smaller(self):
        """Of multiple car boxes, the highest-scoring box is retained even if it is smaller."""
        small_high_score = {
            "class": "car",
            "box": [100, 100, 180, 180],  # area = 6,400
            "score": 0.92,
            "source": "auto",
        }
        large_lower_score = {
            "class": "car",
            "box": [95, 95, 200, 200],  # area = 11,025
            "score": 0.74,
            "source": "auto",
        }
        result = filter_one_car_per_image([small_high_score, large_lower_score])
        self.assertEqual(len(result), 1, "Only one car box must be retained")
        self.assertEqual(result[0]["score"], 0.92, "The highest scoring box must be selected")
        self.assertEqual(result[0]["box"], [100, 100, 180, 180])

        # Test reverse input ordering gives identical result
        result_rev = filter_one_car_per_image([large_lower_score, small_high_score])
        self.assertEqual(len(result_rev), 1)
        self.assertEqual(result_rev[0]["score"], 0.92)

    def test_2_single_most_accurate_car_kept_when_multiple_present(self):
        """Even with multiple candidate cars on an image, exactly one most accurate car is kept."""
        car_1 = {
            "class": "car",
            "box": [50, 50, 150, 150],
            "score": 0.88,
            "source": "auto",
        }
        car_2 = {
            "class": "car",
            "box": [300, 300, 420, 420],
            "score": 0.81,
            "source": "auto",
        }
        car_3 = {
            "class": "car",
            "box": [10, 10, 60, 60],
            "score": 0.55,
            "source": "auto",
        }
        result = filter_one_car_per_image([car_1, car_2, car_3])
        self.assertEqual(len(result), 1, "Exactly one car box must be returned for the image")
        self.assertEqual(result[0]["score"], 0.88, "The single most accurate car box is returned")
        self.assertEqual(result[0]["box"], [50, 50, 150, 150])

    def test_3_equal_score_uses_larger_area_as_tie_breaker(self):
        """Equal-score boxes use larger area as the tie-breaker."""
        box_smaller = {
            "class": "car",
            "box": [100, 100, 180, 180],  # area = 6,400
            "score": 0.85,
            "source": "auto",
        }
        box_larger = {
            "class": "car",
            "box": [90, 90, 190, 190],  # area = 10,000
            "score": 0.85,  # Exactly equal score
            "source": "auto",
        }
        result = filter_one_car_per_image([box_smaller, box_larger])
        self.assertEqual(len(result), 1, "Only one car box should be retained")
        self.assertEqual(result[0]["box"], [90, 90, 190, 190], "Larger area box must be retained on tie")

    def test_4_non_car_classes_preserved_while_car_is_pruned_to_one(self):
        """Non-car classes are preserved while car predictions are pruned to the single best box."""
        person_1 = {
            "class": "person",
            "box": [50, 50, 120, 200],
            "score": 0.90,
            "source": "auto",
        }
        person_2 = {
            "class": "person",
            "box": [52, 52, 118, 198],
            "score": 0.82,
            "source": "auto",
        }
        car_a = {
            "class": "car",
            "box": [200, 200, 350, 300],
            "score": 0.89,
            "source": "auto",
        }
        car_b = {
            "class": "car",
            "box": [205, 202, 348, 298],
            "score": 0.70,
            "source": "auto",
        }
        result = filter_one_car_per_image([person_1, person_2, car_a, car_b])
        # Non-car classes preserved (2 persons) + strictly 1 car (car_a, score 0.89)
        self.assertEqual(len(result), 3)
        car_results = [r for r in result if r["class"] == "car"]
        self.assertEqual(len(car_results), 1)
        self.assertEqual(car_results[0]["score"], 0.89)


if __name__ == "__main__":
    unittest.main()
