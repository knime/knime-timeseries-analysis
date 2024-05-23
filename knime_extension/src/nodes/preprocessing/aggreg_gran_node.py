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
        The following scenarios affect the output schema and the specs are then configured:

        | Scenario                                                   | -> Influence on output schema/specs
        | :--------------------------------------------------------- | :--------------------------------------------------
        | Granularity is week, month, or quarter                     | -> Additional column introduced
        | Input timestamp data type and granularity                  | -> Defines output timestamp data type
        | Numerical input target data type and aggregation method    | -> Defines data type of the numerical output target
        """
        params = self.aggreg_params
        col_names = input_schema.column_names

        input_aggregation_column_ktype = (
            input_schema[:]
            .delegate._columns[col_names.index(params.aggregation_column)]
            .ktype
        )

        input_timestamp_ktype = (
            input_schema[:]
            .delegate._columns[col_names.index(params.datetime_col)]
            .ktype
        )

        timestamp_value_factory_class_string = json.loads(
            input_timestamp_ktype.logical_type
        ).get("value_factory_class")

        target_column_is_of_type_double = False

        if input_aggregation_column_ktype == knext.double():
            if params.aggregation_methods != params.AggregationMethods.COUNT.name:
                target_column_is_of_type_double = True
        else:
            if params.aggregation_methods in [
                params.AggregationMethods.MEAN.name,
                params.AggregationMethods.VARIANCE.name,
            ]:
                target_column_is_of_type_double = True

        contains_timezone = (
            timestamp_value_factory_class_string == kutil.ZONED_DATE_TIME_ZONE_VALUE
            and params.time_granularity
            in [
                params.TimeGranularityOpts.HOUR.name,
                params.TimeGranularityOpts.MINUTE.name,
                params.TimeGranularityOpts.SECOND.name,
            ]
        )

        contains_time = timestamp_value_factory_class_string in [
            kutil.ZONED_DATE_TIME_ZONE_VALUE,
            kutil.LOCAL_DATE_TIME_VALUE,
            kutil.LOCAL_TIME_VALUE,
        ] and (
            params.time_granularity
            in [
                params.TimeGranularityOpts.HOUR.name,
                params.TimeGranularityOpts.MINUTE.name,
                params.TimeGranularityOpts.SECOND.name,
            ]
        )

        contains_date = (
            timestamp_value_factory_class_string
            in [
                kutil.ZONED_DATE_TIME_ZONE_VALUE,
                kutil.LOCAL_DATE_TIME_VALUE,
            ]
            and (
                params.time_granularity
                in [
                    params.TimeGranularityOpts.HOUR.name,
                    params.TimeGranularityOpts.DAY.name,
                    params.TimeGranularityOpts.MINUTE.name,
                    params.TimeGranularityOpts.SECOND.name,
                ]
            )
        ) or (
            timestamp_value_factory_class_string == kutil.LOCAL_DATE_VALUE
            and params.time_granularity == params.TimeGranularityOpts.DAY.name
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
        if params.time_granularity == params.TimeGranularityOpts.YEAR.name:
            aggregation_category = AggregationGranularityParams.AggregationCategory.YEAR
        elif params.time_granularity in [
            params.TimeGranularityOpts.WEEK.name,
            params.TimeGranularityOpts.MONTH.name,
            params.TimeGranularityOpts.QUARTER.name,
        ]:
            aggregation_category = (
                AggregationGranularityParams.AggregationCategory.WEEK_OR_LONGER
            )
        else:
            aggregation_category = (
                AggregationGranularityParams.AggregationCategory.DAY_OR_SHORTER
            )

        return self.__create_schema(
            target_column_is_of_type_double,
            contains_timezone,
            contains_time,
            contains_date,
            aggregation_category,
        )

    def __create_schema(
        self,
        target_column_is_of_type_double: bool,
        contains_timezone: bool,
        contains_time: bool,
        contains_date: bool,
        aggregation_category: AggregationGranularityParams.AggregationCategory,
    ):
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

    def execute(self, exec_context: knext.ExecutionContext, input_1: knext.Table):
        df = input_1.to_pandas()

        params = self.aggreg_params

        date_time_col_orig = df[params.datetime_col]
        agg_col = df[params.aggregation_column]

        # Cast agg_col always to long to avoid casting issues
        if pd.api.types.is_integer_dtype(agg_col):
            agg_col = agg_col.astype(np.int64)
        exec_context.set_progress(0.1)

        kn_date_time_format = kutil.get_type_timestamp(str(date_time_col_orig.dtype))

        if kn_date_time_format == kutil.DEF_ZONED_DATE_LABEL:
            a = kutil.cast_to_related_type(kn_date_time_format, date_time_col_orig)
            date_time_col, kn_date_time_format, zone_offset = a[0], a[1], a[2]

        else:
            a = kutil.cast_to_related_type(kn_date_time_format, date_time_col_orig)
            date_time_col, kn_date_time_format = a[0], a[1]

        exec_context.set_progress(0.5)

        df_time = kutil.extract_time_fields(
            date_time_col, kn_date_time_format, str(date_time_col.name)
        )

        time_granularity = params.time_granularity.capitalize()
        aggreg_method = params.aggregation_methods

        if time_granularity not in df_time.columns:
            raise knext.InvalidParametersError(
                f"""Selected timestamp column does not contain {time_granularity} field."""
            )
        exec_context.set_progress(0.6)

        df_time = self.__modify_time(time_granularity, kn_date_time_format, df_time)

        exec_context.set_progress(0.7)

        if (kn_date_time_format == kutil.DEF_ZONED_DATE_LABEL) and (
            time_granularity in kutil.time_granularity_list()
        ):
            df_time = self.__append_time_zone(df_time, zone_offset)

        df_grouped = self.__aggregate(df_time, agg_col, time_granularity, aggreg_method)

        if pd.api.types.is_integer_dtype(df_grouped[params.aggregation_column]):
            df_grouped[params.aggregation_column] = df_grouped[
                params.aggregation_column
            ].astype(np.int64)

        exec_context.set_progress(0.8)
        if time_granularity not in (
            params.TimeGranularityOpts.QUARTER.name.capitalize(),
            params.TimeGranularityOpts.MONTH.name.capitalize(),
            params.TimeGranularityOpts.WEEK.name.capitalize(),
        ):
            df_grouped = df_grouped[[params.datetime_col, params.aggregation_column]]
        exec_context.set_progress(0.9)

        return knext.Table.from_pandas(df_grouped)

    def __modify_time(self, time_gran: str, kn_date_time_type: str, df: pd.DataFrame):
        """
        This function modifies the input timestamp column according to the type of granularity selected. For instance, if the selected time granularity is "Quarter" then
        the next higher time value against quarter will be "Year". Hence only "Year" will be returned.
        """

        if kn_date_time_type == kutil.DEF_TIME_LABEL:
            date_col = pd.to_datetime(
                df[self.aggreg_params.datetime_col], format=kutil.TIME_FORMAT
            ).dt.strftime("%H:%M:%S")
        else:
            date_col = df[self.aggreg_params.datetime_col].astype("datetime64[ns]")

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
        Floor the timestamp against the selected time granularity.
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
        Final aggregation based on the selected level of granularity in given datetime column.
        The aggregation is done on the modified date column and the datetime field corresponding to the selected granularity.
        """

        df = pd.concat([df_time, aggregation_column], axis=1)

        df = df[
            [
                self.aggreg_params.datetime_col,
                self.aggreg_params.aggregation_column,
                time_gran,
            ]
        ]
        granularity = self.aggreg_params.AggregationDictionary[agg_type.upper()].value[
            1
        ]

        df = (
            df.groupby([df[self.aggreg_params.datetime_col], time_gran])
            .agg(granularity)
            .reset_index()
        )

        return df

    def __append_time_zone(self, date_col: pd.DataFrame, zoned: pd.Series):
        """
        This function re-assignes time zones to date&time column. This function is only called if input date&time column containes time zone.
        """
        # TODO Find faster approach # NOSONAR not urgent
        date_col_internal = date_col
        for i in range(0, len(zoned.index)):
            date_col_internal[self.aggreg_params.datetime_col][i] = date_col_internal[
                self.aggreg_params.datetime_col
            ][i].replace(tzinfo=zoned.iloc[i])
        return date_col_internal
