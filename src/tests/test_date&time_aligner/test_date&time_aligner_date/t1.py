import unittest
from nodes.preprocessing.timestamp_alignment_node import TimestampAlignmentNode

class TestTimeStampAlignmentDate(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_sample_zero(self):

        self.assertTrue(1 == 1, "Sample test, relax!")

    def test_sample_first(self):

        self.assertTrue(1 == 1, "Sample test, relax!")

    def test_sample_second(self):

        self.assertTrue(1 == 1, "Sample test, relax!")

    def test_sample_third(self):

        self.assertTrue(1 == 1, "Sample test, relax!")

if __name__ == "__main__":
    unittest.main()


