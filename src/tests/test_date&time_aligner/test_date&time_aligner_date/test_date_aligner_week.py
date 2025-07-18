import os
import sys
import unittest
import knime.extension as knext
import knime.extension.testing as ktest
import pandas as pd
import datetime
import pathlib
import debugpy
debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()


# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from nodes.preprocessing.timestamp_alignment_node import TimestampAlignmentNode
from nodes.configs.preprocessing.timealign import TimeStampAlignmentParams


class TestTimeStampAlignmentDate(unittest.TestCase):
    test_date_df = pd.read_csv(
        str(pathlib.Path(__file__).parent.parent.parent / "data" / "Date_Test.csv")
    )

    @classmethod
    def setUpClass(self) -> None:
        super().setUpClass()

        import requests

        def download(url, file):
            r = requests.get(url, allow_redirects=True)
            if os.path.exists(file):
                os.remove(file)
            with open(file, "wb") as f:
                f.write(r.content)

        download(
            "https://raw.githubusercontent.com/knime/knime-python/refs/heads/master/org.knime.python3.arrow.types/plugin.xml",
            "plugin.xml",
        )
        # src/main/python/knime/types is the location of the source file that contains type mappings
        os.makedirs("src/main/python/knime/types", exist_ok=True)
        download(
            "https://raw.githubusercontent.com/knime/knime-python/refs/heads/master/org.knime.python3.arrow.types/src/main/python/knime/types/builtin.py",
            "src/main/python/knime/types/builtin.py",
        )
        # {"org.knime.core.data.v2.time.LocalDateValueFactory":"datetime.date"}
        ktest.register_extension(
            "plugin.xml",
            {"org.knime.core.data.v2.time.LocalDateValueFactory": "datetime.date"},
        )
        # raise ValueError(knext.LogicalType.supported_value_types())

        self.input_schema = knext.Schema.from_columns(
            [
                knext.Column(knext.double(), "C15"),
                knext.Column(knext.int32(), "C16"),
                knext.Column(knext.double(), "C17"),
                knext.Column(knext.logical(datetime.date), "date"),
                knext.Column(knext.double(), "C19"),
                knext.Column(knext.double(), "C20"),
            ]
        )

        self.expected_output_schema_columns = [
                knext.Column(knext.double(), "C15"),
                knext.Column(knext.int32(), "C16"),
                knext.Column(knext.double(), "C17"),
                knext.Column(knext.logical(datetime.date), "date"),
                knext.Column(knext.double(), "C19"),
                knext.Column(knext.double(), "C20"),
            ]

    @classmethod
    def tearDownClass(cls):
        os.remove("plugin.xml")
        os.remove("src/main/python/knime/types/builtin.py")
        return super().tearDownClass()

    def _get_dataframe(self):
        return self.test_date_df

    def _get_column_names_from_test_data(self):
        return self.test_date_df.columns

    def test_node_configure(self):
        self.setUp()
        node = TimestampAlignmentNode()
        config_context = ktest.TestingConfigurationContext()

        # configure node
        node.params.datetime_col = "date"

        node.replace_original = True
        node.params.period = TimeStampAlignmentParams.Period.WEEK.name

        output_schema = node.configure(config_context, self.input_schema)

        expected_schema = knext.Schema.from_columns(self.expected_output_schema_columns)
        self.assertEqual(expected_schema, output_schema)

    # this test function checks that the output schema is correct
    # when the pandas table is sent back to knime.
    def test_table_schema(self):
        input_df = self._get_dataframe()

        input_df["date"] = pd.to_datetime(input_df["date"], format="%Y-%m-%d").dt.date
        print(input_df["date"].dtypes)

        table = knext.Table.from_pandas(input_df)
        print("Table Schema", f"={table.schema}",)

        expected_schema = knext.Schema.from_columns(self.expected_output_schema_columns)
        print("Expected schema", f"={expected_schema}")

        self.assertEqual(expected_schema, table.schema)


if __name__ == "__main__":
    unittest.main()


# object_testing = TestTimeStampAlignmentDate()
# print(object_testing._get_column_names_from_test_data())
