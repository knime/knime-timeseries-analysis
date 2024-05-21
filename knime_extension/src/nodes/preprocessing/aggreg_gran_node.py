import logging
import json
import knime.extension as knext
from util import utils as kutil
import pandas as pd
import numpy as np
from ..configs.preprocessing.aggrgran import (
    AggregationGranularityParams,
)

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Date&Time Aggregator (Labs)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/preprocessing/Aggregation_Granularity.png",
    category=kutil.category_processsing,
    id="aggregation_granularity",
)
@knext.input_table(
    name="Input Data",
    description="Table containing the timestamp column and the numeric column to apply the selected aggregation method",
)
@knext.output_table(
    name="Aggregated Output",
    description="Table output containing the timestamp based on selected granularity and the aggregated numerical column.",
)
class AggregationGranularity:
    """
    Aggregate data based on a timestamp column and selected granularity: minute, hour, day, week, month, quarter, year.

    The aggregation granularity node works a lot like group by node except to define your grouping, select a timestamp column and a level of granularity to define your groups. Then numeric columns can be aggregated to: mode, min, max, sum, var, count, or mean.
    *mode will return the first mode in the event of a tie.*
    """

    aggreg_params = AggregationGranularityParams()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        # TODO Specify the output schema which depends on the selected parameters # NOSONAR

        # set date&time column by default
        self.aggreg_params.datetime_col = kutil.column_exists_or_preset(
            configure_context,
            self.aggreg_params.datetime_col,
            input_schema_1,
            kutil.is_type_timestamp,
        )

        # set aggregation column
        self.aggreg_params.aggregation_column = kutil.column_exists_or_preset(
            configure_context,
            self.aggreg_params.aggregation_column,
            input_schema_1,
            kutil.is_numeric,
        )

        return self.__configure_specs(input_schema_1)

    def __configure_specs(self, input_schema: knext.Schema) -> knext.Schema:
        """
        This function runs the logic of all possible combinations that can produce output schema contianing:

        1) 2 or 3 columns.
        2) The timestamp data type after performing the aggregation upon the the selected dattime field.
        3) The possible datatype of the target column (column on which aggregation method is applied). This can be either int64 or double.
        """
        # initialize the variable that is needed for returning the output schema and get the column names
        output_schema = None
        col_names = input_schema.column_names

        # get data type of the target column ==> double or int
        input_aggregation_column_ktype = (
            input_schema[:]
            .delegate._columns[col_names.index(self.aggreg_params.aggregation_column)]
            .ktype
        )

        # get knime data type of the selected timestamp column
        input_timestamp_ktype = (
            input_schema[:]
            .delegate._columns[col_names.index(self.aggreg_params.datetime_col)]
            .ktype
        )

        # get the value factory string of the respective datetime type;
        # eg. ZonedDateTimeValueFactory2, LocalDateTimeValueFactory, LocalDateValueFactory and LocalTimeValueFactory
        time_stamp_logical_type = json.loads(input_timestamp_ktype.logical_type).get(
            "value_factory_class"
        )

        # this variable decides if the target column will be a double or an int64.
        # Unless the selected method is not Count then the output can be a double or int64 based on the data type of input target column.
        target_column_is_of_type_double = False

        if input_aggregation_column_ktype == knext.double():
            if (
                self.aggreg_params.aggregation_methods
                != self.aggreg_params.AggregationMethods.COUNT.name
            ):
                target_column_is_of_type_double = True
        else:
            if self.aggreg_params.aggregation_methods in [
                self.aggreg_params.AggregationMethods.MEAN.name,
                self.aggreg_params.AggregationMethods.VARIANCE.name,
            ]:
                target_column_is_of_type_double = True

        # only if the input timestamp data type is of type Zoned Date&Time and the selected time granularity is in hours, minutes and seconds
        # then the output datatype of timestamp column will always have a timezone.
        contains_timezone = (
            time_stamp_logical_type == kutil.ZONED_DATE_TIME_ZONE_VALUE
            and self.aggreg_params.time_granularity
            in [
                self.aggreg_params.TimeGranularityOpts.HOUR.name,
                self.aggreg_params.TimeGranularityOpts.MINUTE.name,
                self.aggreg_params.TimeGranularityOpts.SECOND.name,
            ]
        )

        # if the input timestamp data type contains a time value, AND IF upon the selected aggregation of time granularity is in hour, minutes or seconds, then only the output will contain a time.
        contains_time = time_stamp_logical_type in [
            kutil.ZONED_DATE_TIME_ZONE_VALUE,
            kutil.LOCAL_DATE_TIME_VALUE,
            kutil.LOCAL_TIME_VALUE,
        ] and (
            self.aggreg_params.time_granularity
            in [
                self.aggreg_params.TimeGranularityOpts.HOUR.name,
                self.aggreg_params.TimeGranularityOpts.MINUTE.name,
                self.aggreg_params.TimeGranularityOpts.SECOND.name,
            ]
        )

        # the boolean flag for this field is decided in two ways:
        ## if the input timestamp data type contains a date AND if the time granularity is in Hour, Day, Minute or Seconds, then the output data type of timestamp column will contain a date.
        ## if the input timestamp data type is of type DATE and the selected time granularity is in Day then the output timestamp data type will always contain a date.
        contains_date = (
            time_stamp_logical_type
            in [
                kutil.ZONED_DATE_TIME_ZONE_VALUE,
                kutil.LOCAL_DATE_TIME_VALUE,
            ]
            and (
                self.aggreg_params.time_granularity
                in [
                    self.aggreg_params.TimeGranularityOpts.HOUR.name,
                    self.aggreg_params.TimeGranularityOpts.DAY.name,
                    self.aggreg_params.TimeGranularityOpts.MINUTE.name,
                    self.aggreg_params.TimeGranularityOpts.SECOND.name,
                ]
            )
        ) or (
            time_stamp_logical_type == kutil.LOCAL_DATE_VALUE
            and self.aggreg_params.time_granularity
            == self.aggreg_params.TimeGranularityOpts.DAY.name
        )

        # this option toggles the output schema in two scenarios:
        ## - if the selected time granulariy is "YEAR" then the output schema will contain two columns: a timestamp column containing only YEAR of type int32 and aggregation of type int64.
        ## - if the selected time granualrity is in "WEEK", "MONTH" or "QUARTER", then the output schema will contain three columns:
        ### a timestamp column of "YEAR" of type int32,
        ### a column containing int32 value of selcted time granularity
        ### and finally the target column on which the aggregation method is applied.
        ## if the selected time granularity is "DAY", "HOUR", "MINUTE" and "SECOND" then the output schema will contain two columns:
        ### one will contain a timestamp column of corresponding timestamp data type.
        ### and target column after the aggregation method is applied.
        if (
            self.aggreg_params.time_granularity
            == self.aggreg_params.TimeGranularityOpts.YEAR.name
        ):
            aggregation_category = AggregationGranularityParams.AggregationCategory.YEAR
        elif self.aggreg_params.time_granularity in [
            self.aggreg_params.TimeGranularityOpts.WEEK.name,
            self.aggreg_params.TimeGranularityOpts.MONTH.name,
            self.aggreg_params.TimeGranularityOpts.QUARTER.name,
        ]:
            aggregation_category = (
                AggregationGranularityParams.AggregationCategory.WEEK_OR_LONGER
            )
        else:
            aggregation_category = (
                AggregationGranularityParams.AggregationCategory.DAY_OR_SHORTER
            )

        ## the fnction returns the final output schema
        output_schema = self.__create_schema(
            target_column_is_of_type_double,
            contains_timezone,
            contains_time,
            contains_date,
            aggregation_category,
        )

        return output_schema

    def __create_schema(
        self,
        target_column_is_of_type_double: bool,
        contains_timezone: bool,
        contains_time: bool,
        contains_date: bool,
        aggregation_category: AggregationGranularityParams.AggregationCategory,
    ):
        """
        This function generates the output schema based on the checks that are applied beforehand.
        """

        # specify first column
        if (
            aggregation_category
            is not AggregationGranularityParams.AggregationCategory.DAY_OR_SHORTER
        ):
            data_types = [knext.int32()]
        else:
            data_types = [
                knext.datetime(
                    date=contains_date,
                    time=contains_time,
                    timezone=contains_timezone,
                )
            ]

        column_names = [self.aggreg_params.datetime_col]

        # specify optional second column
        if (
            aggregation_category
            == AggregationGranularityParams.AggregationCategory.WEEK_OR_LONGER
        ):
            data_types.append(knext.int32())
            column_names.append(self.aggreg_params.time_granularity.capitalize())

        # specify third column
        data_types.append(
            knext.double() if target_column_is_of_type_double else knext.int64()
        )
        column_names.append(self.aggreg_params.aggregation_column)

        return knext.Schema(data_types, column_names)

    def execute(self, exec_context: knext.ExecutionContext, input_1: knext.Schema):
        df = input_1.to_pandas()

        date_time_col_orig = df[self.aggreg_params.datetime_col]
        agg_col = df[self.aggreg_params.aggregation_column]

        # cast to long type, this is done to reduce the complication of worrying about if the output specs of target column is weather int32 and int64.
        if pd.api.types.is_integer_dtype(agg_col):
            agg_col = agg_col.astype(np.int64)
        exec_context.set_progress(0.1)

        # get timestamp data type
        kn_date_time_format = kutil.get_type_timestamp(str(date_time_col_orig.dtype))

        # if condition to handle zoned date&time
        if kn_date_time_format == kutil.DEF_ZONED_DATE_LABEL:
            a = kutil.cast_to_related_type(kn_date_time_format, date_time_col_orig)

            date_time_col, kn_date_time_format, zone_offset = a[0], a[1], a[2]

        else:
            # returns series of date time according to the date format and knime supported data type
            a = kutil.cast_to_related_type(kn_date_time_format, date_time_col_orig)

            # handle multiple iterable error. This is done to handle dynamic assignment of variables in case zoned date and time type is encountered
            date_time_col, kn_date_time_format = a[0], a[1]

        exec_context.set_progress(0.5)
        # extract date&time fields from the input timestamp column
        df_time = kutil.extract_time_fields(
            date_time_col, kn_date_time_format, str(date_time_col.name)
        )

        # this variable is assigned the time granularity selected by the user
        selected_time_granularity = self.aggreg_params.time_granularity.capitalize()

        # this variable is assigned the aggregation method selected by the user
        selected_aggreg_method = self.aggreg_params.aggregation_methods

        # raise exception if selected time granularity does not exists in the input timestamp column
        if selected_time_granularity not in df_time.columns:
            raise knext.InvalidParametersError(
                f"""Selected timestamp column does not contain {selected_time_granularity} field."""
            )
        exec_context.set_progress(0.6)

        # modify the input timestamp as per the time_gran selected. This modifies the timestamp column depending on the granularity selected
        df_time_updated = self.__modify_time(
            selected_time_granularity, kn_date_time_format, df_time
        )

        exec_context.set_progress(0.7)

        # if kn_date_time_format contains zone and if selected time granularity is less than day then append the zone back, other wise ignore
        if (kn_date_time_format == kutil.DEF_ZONED_DATE_LABEL) and (
            selected_time_granularity in kutil.time_granularity_list()
        ):
            df_time_updated = self.__append_time_zone(df_time_updated, zone_offset)

        # perform final aggregation
        df_grouped = self.__aggregate(
            df_time_updated, agg_col, selected_time_granularity, selected_aggreg_method
        )

        if pd.api.types.is_integer_dtype(
            df_grouped[self.aggreg_params.aggregation_column]
        ):
            df_grouped[self.aggreg_params.aggregation_column] = df_grouped[
                self.aggreg_params.aggregation_column
            ].astype(np.int64)

        exec_context.set_progress(0.8)
        if selected_time_granularity not in (
            self.aggreg_params.TimeGranularityOpts.QUARTER.name.capitalize(),
            self.aggreg_params.TimeGranularityOpts.MONTH.name.capitalize(),
            self.aggreg_params.TimeGranularityOpts.WEEK.name.capitalize(),
        ):
            df_grouped = df_grouped[
                [self.aggreg_params.datetime_col, self.aggreg_params.aggregation_column]
            ]
        exec_context.set_progress(0.9)

        return knext.Table.from_pandas(df_grouped)

    def __modify_time(
        self, time_gran: str, kn_date_time_type: str, df_time: pd.DataFrame
    ):
        """
        This function modifies the input timestamp column according to the type of granularity selected. For instance, if the selected time granularity is "Quarter" then
        the next higher time value against quarter will be "Year". Hence only "Year" will be returned.
        """

        df = df_time.copy()

        if kn_date_time_type == kutil.DEF_TIME_LABEL:
            date_col = pd.to_datetime(
                df[self.aggreg_params.datetime_col], format=kutil.TIME_FORMAT
            ).dt.strftime("%H:%M:%S")
        else:
            date_col = df[self.aggreg_params.datetime_col].astype("datetime64[ns]")

        # check if granularity level is
        if time_gran in (
            self.aggreg_params.TimeGranularityOpts.YEAR.name.capitalize(),
            self.aggreg_params.TimeGranularityOpts.QUARTER.name.capitalize(),
            self.aggreg_params.TimeGranularityOpts.MONTH.name.capitalize(),
            self.aggreg_params.TimeGranularityOpts.WEEK.name.capitalize(),
        ):
            # return year only
            date_col = date_col.dt.year
            df[self.aggreg_params.datetime_col] = date_col
            df[self.aggreg_params.datetime_col] = df[
                self.aggreg_params.datetime_col
            ].astype(np.int32)

        # set input timestamp to date_col
        elif time_gran == self.aggreg_params.TimeGranularityOpts.DAY.name.capitalize():

            date_col = date_col.dt.date

            df[self.aggreg_params.datetime_col] = date_col

        # round datetime to nearest hour
        elif time_gran == self.aggreg_params.TimeGranularityOpts.HOUR.name.capitalize():
            df[self.aggreg_params.datetime_col] = self.__floor_time(
                kn_date_time_type, "H", date_col
            )

        # round datetime to nearest minute
        elif (
            time_gran == self.aggreg_params.TimeGranularityOpts.MINUTE.name.capitalize()
        ):
            df[self.aggreg_params.datetime_col] = self.__floor_time(
                kn_date_time_type, "min", date_col
            )

        # round datetime to nearest second. This option is feasble if timestamp contains milliseconds/microseconds/nanoseconds.
        elif (
            time_gran == self.aggreg_params.TimeGranularityOpts.SECOND.name.capitalize()
        ):
            df[self.aggreg_params.datetime_col] = self.__floor_time(
                kn_date_time_type, "S", date_col
            )

        return df

    def __floor_time(
        self, kn_date_time_type: str, time_gran: str, date: pd.Series
    ) -> pd.Series:
        """
        This function is use to floor the timestamp against the selected time granularity.
        """

        if kn_date_time_type == kutil.DEF_TIME_LABEL:
            date = pd.to_datetime(date, format=kutil.TIME_FORMAT)
            date = date.dt.floor(time_gran)
            date = date.dt.time

        else:
            date = pd.to_datetime(date, format=kutil.DATE_TIME_FORMAT)
            date = date.dt.floor(time_gran)

        return date

    def __aggregate(
        self,
        df_time: pd.Series,
        aggregation_column: pd.Series,
        time_gran: str,
        agg_type: str,
    ):
        """
        This function performs the final aggregation based on the selected level of granularity in given datetime column.
        The aggregation is done on the modified date column and the datetime field corresponding to the selected granularity.

        """

        # pre-process
        df = pd.concat([df_time, aggregation_column], axis=1)

        # filter out necessary columns
        df = df[
            [
                self.aggreg_params.datetime_col,
                self.aggreg_params.aggregation_column,
                time_gran,
            ]
        ]
        # select granularity
        value = self.aggreg_params.AggregationDictionary[agg_type.upper()].value[1]

        # aggregate for final output
        df = (
            df.groupby([df[self.aggreg_params.datetime_col], time_gran])
            .agg(value)
            .reset_index()
        )

        return df

    def __append_time_zone(self, date_col: pd.DataFrame, zoned: pd.Series):
        """
        This function re-assignes time zones to date&time column. This function is only called if input date&time column containes time zone.
        """

        date_col_internal = date_col
        for i in range(0, len(zoned.index)):
            date_col_internal[self.aggreg_params.datetime_col][i] = date_col_internal[
                self.aggreg_params.datetime_col
            ][i].replace(tzinfo=zoned.iloc[i])
        return date_col_internal
