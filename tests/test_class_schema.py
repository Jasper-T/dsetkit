import unittest

from dsetkit import compare_class_schemas


class TestCompareClassSchemas(unittest.TestCase):
    def test_strict_accepts_identical_ids_names_and_order(self) -> None:
        result = compare_class_schemas(["cat", "dog"], ["cat", "dog"])

        self.assertTrue(result.compatible)
        self.assertEqual(result.differences, ())

    def test_strict_reports_order_and_length_differences(self) -> None:
        result = compare_class_schemas(
            ["cat", "dog", "person"],
            ["dog", "cat"],
        )

        self.assertFalse(result.compatible)
        self.assertEqual(
            [difference.reason for difference in result.differences],
            ["name_mismatch", "name_mismatch", "missing_on_right"],
        )
        self.assertEqual(result.differences[0].class_id, 0)

    def test_rejects_unsupported_mode(self) -> None:
        with self.assertRaises(ValueError):
            compare_class_schemas(["cat"], ["cat"], mode="by_name")


if __name__ == "__main__":
    unittest.main()
