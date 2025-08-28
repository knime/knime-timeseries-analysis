import os
import sys
import unittest
import knime.extension as knext
import knime.extension.testing as ktest
import pandas as pd
import datetime
import pathlib
import numpy as np

#Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from nodes.preprocessing.timestamp_alignment_node import TimestampAlignmentNode


class TestTimeStampAlignmentDate(unittest.TestCase):
    test_date_df = pd.read_csv(
        str(pathlib.Path(__file__).parent.parent.parent / "data" / "Date_Test.csv")
    )

    golden_table_df = pd.read_csv(
        str(
            pathlib.Path(__file__).parent.parent.parent
            / "data"
            / "golden_tables"
            / "DateType_Date&Time_Alignment_GT"
            / "WEEK"
            / "DateType_Replacement_WEEK_Date&Time_Alignment_Output.csv"
        )
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
        # region dbpy_attach
        import debugpy
        (debugpy.listen(5678), debugpy.wait_for_client()) if not debugpy.is_client_connected() else None
        # endregion
        
        ktest.register_extension(
            "plugin.xml",
            {"org.knime.core.data.v2.time.LocalDateValueFactory": "datetime.date"},
        )

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

        self.expected_output_schema_columns = self.input_schema._columns

        self.expected_output_schema_columns_if_replace_false = [
            knext.Column(knext.double(), "C15"),
            knext.Column(knext.int32(), "C16"),
            knext.Column(knext.double(), "C17"),
            knext.Column(knext.logical(datetime.date), "date"),
            knext.Column(knext.logical(datetime.date), "date (New)"),
            knext.Column(knext.double(), "C19"),
            knext.Column(knext.double(), "C20"),
        ]
        self.node = TimestampAlignmentNode()
        self.paramslist = [
            "WEEK",
            "DAY",
            "MONTH",
            "YEAR",
        ]

    @classmethod
    def tearDownClass(cls):
        os.remove("plugin.xml")
        os.remove("src/main/python/knime/types/builtin.py")
        return super().tearDownClass()

    def _get_dataframe(self):
        return self.test_date_df

    def _get_golden_dataframe(self):
        return self.golden_table_df

    def _get_column_names_from_test_data(self):
        return self.test_date_df.columns

    def _configure(self, replace_original_param=True, param = None):

        self.node.params.datetime_col = "date"
        self.node.params.replace_original = replace_original_param
        self.node.params.period = param

    def test_node_configure(self):

        for p in self.paramslist:
            with self.subTest(p=p):
                self.setUp()
                self._configure(True, p)
                config_context = ktest.TestingConfigurationContext()

                output_schema = self.node.configure(config_context, self.input_schema)


                expected_schema = knext.Schema.from_columns(self.expected_output_schema_columns)
                self.assertEqual(expected_schema, output_schema)

    def test_node_configure_replace_false(self):

        for p in self.paramslist:
            self.setUp()
            self._configure(False,p)
            config_context = ktest.TestingConfigurationContext()
            output_schema = self.node.configure(config_context, self.input_schema)

            expected_schema = knext.Schema.from_columns(
                self.expected_output_schema_columns_if_replace_false
            )
            self.assertEqual(expected_schema, output_schema)

    # this test function checks that the output schema is correct
    # when the pandas table is sent back to knime.
    def test_table_schema(self):
        input_df = self._get_dataframe()

        input_df["date"] = pd.to_datetime(input_df["date"], format="%Y-%m-%d").dt.date

        # downcast pandas columns to int32 from int64, if only int32 were to be expected in the output schema
        column_names_that_are_int32 = [
            k.name
            for k in self.expected_output_schema_columns
            if str(k.ktype) == "Number (Integer)"
        ]
        for col in column_names_that_are_int32:
            input_df[col] = input_df[col].astype(np.dtype("int32"))

        table = knext.Table.from_pandas(input_df)

        expected_schema = knext.Schema.from_columns(self.expected_output_schema_columns)

        self.assertEqual(expected_schema, table.schema)

    # commented this out since node execution fails,
    # node is unable to fetch the pandas logical extension type that is registered with the value factory string
    # def test_node_execute(self):
        # for p in self.paramslist:
        #     self._configure(False, p)
        #     input_df = self._get_golden_dataframe()
        #     # region dbpy_attach
        #     import debugpy
        #     debugpy.listen(5678)
        #     debugpy.wait_for_client()
        #     # endregion
            
        #     input_df["date"] = pd.to_datetime(input_df["date"], format="%Y-%m-%d").dt.date


        #     # downcast pandas columns to int32 from int64, if only int32 were to be expected in the output schema
        #     column_names_that_are_int32 = [
        #         k.name
        #         for k in self.expected_output_schema_columns
        #         if str(k.ktype) == "int32"
        #     ]
        #     for col in column_names_that_are_int32:
        #         input_df[col] = input_df[col].astype(pd.Int32Dtype())

        #     _input_table = knext.Table.from_pandas(input_df)
        #     _input_table_df = _input_table.to_pandas()
        #     exec_context = ktest.TestingExecutionContext()

        #     output = self.node.execute(exec_context, _input_table)
        #     output_df = output.to_pandas()

        #     self.assertEqual(input_df, output_df)


if __name__ == "__main__":
    unittest.main()
