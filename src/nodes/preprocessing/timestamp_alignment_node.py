import logging
import knime.extension as knext
from util import utils as kutil
import pandas as pd
from ..configs.preprocessing.timealign import TimeStampAlignmentParams

NEW_COLUMN = " (New)"

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Date&Time Aligner (Labs)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/preprocessing/Timestamp_Alignment.png",
    category=kutil.category_processsing,
    id="timestamp_alignment",
)
@knext.input_table(
    name="Input Data",
    description="Table containing the timestamp column to fill in for the missing timestamps based on the time granularity selected.",
)
@knext.output_table(
    name="Output Data",
    description="Table output with the column with missing timestamps in the given range.",
)
class TimestampAlignmentNode:
    """
    Checks a table for non-existent timestamps and generates rows with missing values for them.

    Select a timestamp column and a time granularity. The node will verify that a record exists in your table for each value at that granularity, for example if you select hours it will check for 01:00, 02:00, 03:00… if a timestamp is not found it will be inserted and missing values generated for the remaining columns.
    The final output is sorted in ascending order of newly generated timestsamp.
    Use this in combination with the missing value node to correct missing time series data.
    This node preserves duplicated values and possibly lead to cluttered output. Therefore, in case of duplicated timestamps, we encourage using *Date&Time Aggregator* node before using this node.
    """

    params = TimeStampAlignmentParams()

    def configure(
        self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema
    ):
        self.params.datetime_col = kutil.column_exists_or_preset(
            configure_context,
            self.params.datetime_col,
            input_schema,
            kutil.is_type_timestamp,
        )

        date_ktype = input_schema[[self.params.datetime_col]].delegate._columns[input_schema.column_names.index(self.params.datetime_col)].ktype

        if not self.params.replace_original:
            datetime_index = (
                input_schema.column_names.index(self.params.datetime_col) + 1
            )
            input_schema = input_schema.insert(
                knext.Column(
                    date_ktype,
                    self.params.datetime_col + NEW_COLUMN,
                ),
                datetime_index,
            )
        return input_schema

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        df = input_table.to_pandas()

        datetime_col = df[self.params.datetime_col]

        self.__validate(datetime_col)

        # 'knime.pandas_type<int64, {"value_factory_class":"org.knime.core.data.v2.time.LocalTimeValueFactory"}>'
        print(str(datetime_col.dtype))

        timestamp_value_factory_class_string = kutil.get_type_timestamp(
            str(datetime_col.dtype)
        )
        selected_period = self.params.period.capitalize()
        exec_context.set_progress(0.2)

        zone_offset = None
        if timestamp_value_factory_class_string == kutil.DEF_ZONED_DATE_LABEL:
            a = kutil.cast_to_related_type(
                timestamp_value_factory_class_string, datetime_col
            )
            date_time_col, timestamp_value_factory_class_string, zone_offset = (
                a[0],
                a[1],
                a[2],
            )

        else:
            a = kutil.cast_to_related_type(
                timestamp_value_factory_class_string, datetime_col
            )
            date_time_col, timestamp_value_factory_class_string = a[0], a[1]
        exec_context.set_progress(0.3)

        df_time = kutil.extract_time_fields(
            date_time_col, timestamp_value_factory_class_string, str(date_time_col.name)
        )

        if selected_period not in df_time.columns:
            raise knext.InvalidParametersError(
                f"""Input timestamp column cannot resample on {selected_period} field. Please change timestamp data type and try again."""
            )
        # following dataframe does contains both old date column and newly populated date column
        df_time = self.__modify_time(
            timestamp_value_factory_class_string, df_time, zone_offset
        )
        exec_context.set_progress(0.4)

        df = (
            df_time.merge(df, how="left", left_index=True, right_index=True)
            .reset_index(drop=True)
            .sort_values(self.params.datetime_col + NEW_COLUMN)
            .reset_index(drop=True)
        )
        exec_context.set_progress(0.7)

        if self.params.replace_original:
            df = df.drop(columns=[self.params.datetime_col]).rename(
                columns={
                    self.params.datetime_col + NEW_COLUMN: self.params.datetime_col
                }
            )
        exec_context.set_progress(0.9)

        #get specs from configure and ensure that columns in dataframe are in the same order.
        if not self.params.replace_original:

            datetime_index = (
                input_table.schema.column_names.index(self.params.datetime_col) + 1
            )

            # get shallow copy of the list of column names and insert the new datetime column
            # next to the index of the original datetime column
            new_list = input_table.schema.column_names.copy()
            new_list.insert(datetime_index, self.params.datetime_col + NEW_COLUMN)

            df = df.loc[:, new_list]

        # if Replace timestamp column is true, then we need to ensure that the columns in the dataframe are in the same order as the input table schema
        else:
            
            df = df.loc[:, input_table.schema.column_names]

        # added a patch that ensures that columns that are int32 STAY int32 when sent back to KNIME
        column_names_that_are_int32 = [k.name for k in input_table._schema._columns if str(k.ktype) == "Number (integer)"]

        for col in column_names_that_are_int32:
            df[col] = df[col].astype(pd.Int32Dtype())

        return knext.Table.from_pandas(df)

    def __modify_time(
        self, kn_date_format: str, df_time: pd.DataFrame, tz=None
    ) -> pd.DataFrame:
        """
        This function is where the date column is processed to fill in for missing time stamp values
        """

        start = df_time[self.params.datetime_col].astype(str).min()
        end = df_time[self.params.datetime_col].astype(str).max()
        frequency = self.params.TimeFrequency[self.params.period].value[1]

        timestamps = pd.date_range(start=start, end=end, freq=frequency)

        if kn_date_format == kutil.DEF_TIME_LABEL:
            timestamps = pd.Series(timestamps.time)
            modified_dates = self.__align_time(timestamps=timestamps, df=df_time)

        elif kn_date_format == kutil.DEF_DATE_LABEL:
            timestamps = pd.to_datetime(pd.Series(timestamps), format=kutil.DATE_FORMAT)
            timestamps = timestamps.dt.date

            modified_dates = self.__align_time(timestamps=timestamps, df=df_time)

        elif kn_date_format == kutil.DEF_DATE_TIME_LABEL:
            timestamps = pd.to_datetime(
                pd.Series(timestamps), format=kutil.DATE_TIME_FORMAT
            )

            modified_dates = self.__align_time(timestamps=timestamps, df=df_time)

        elif kn_date_format == kutil.DEF_ZONED_DATE_LABEL:
            # region dbpy_attach
            import debugpy
            (debugpy.listen(5678), debugpy.wait_for_client()) if not debugpy.is_client_connected() else None
            # endregion
            
            unique_tz = pd.unique(tz)

            LOGGER.warning("Timezone(s) in the column:" + str(unique_tz))

            if len(unique_tz) > 1:
                raise knext.InvalidParametersError(
                    "Selected date&time column contains multiple zones."
                )
            else:
                modified_dates = self.__align_time(timestamps=timestamps, df=df_time)
                for column in modified_dates.columns:
                    modified_dates[column] = modified_dates[column].dt.tz_localize(unique_tz[0], ambiguous = True, nonexistent='shift_forward')

        return modified_dates

    def __align_time(self, timestamps: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a new column in the existing dataframe, by doing left join on the processed column with the table.
        """
        __duplicate = "_Dup12345"

        # find set difference from available timestamps and missing timestamps
        df2 = pd.DataFrame(
            set(timestamps).difference(df[self.params.datetime_col]),
            columns=[self.params.datetime_col + __duplicate],
        )

        df2 = df2.set_index(self.params.datetime_col + __duplicate, drop=False)      
        # concatenate differenced timestamps with input timestamps
        df3 = pd.DataFrame(
            pd.concat(
                [
                    df[self.params.datetime_col],
                    df2[self.params.datetime_col + __duplicate],
                ]
            )
        ).rename(columns={0: self.params.datetime_col + NEW_COLUMN})

        # do a left join and return only the actual time input and updated timestamp column
        final_df = df3.merge(  # NOSONAR 'on' and 'validate'  do not really need do be specified here
            df, how="left", left_index=True, right_index=True, sort=True
        )

        return final_df[
            [
                self.params.datetime_col + NEW_COLUMN,
            ]
        ]
    
    def __validate(self, timestamp_column: pd.Series):
        """
        Validate the timestamp column to ensure remaining execution do not break.
        """
        
        # ideally an option "Skip missing values" should be added to the node configuration,
        # so that the node can skip missing values in the timestamp column and not break.
        # but since it is not available, we prompt user to clean the data before proceeding.
        if kutil.check_missing_values(timestamp_column):
            raise ValueError(
                "The selected timestamp column contains missing values. Please clean the data before proceeding."
            )
