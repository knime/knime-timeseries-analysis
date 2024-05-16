import logging
import json
import knime.extension as knext
from util import utils as kutil
import pandas as pd
import numpy as np
from ..configs.preprocessing.aggrgran import AggregationGranularityParams
from enum import Enum

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
        # import debugpy

        # debugpy.listen(5678)
        # print("Waiting for debugger attach")
        # debugpy.wait_for_client()
        # debugpy.breakpoint()

        return self.__configure_specs(input_schema_1)

    def __configure_specs(self, input_schema: knext.Schema) -> knext.Schema:
        output_schema = None
        col_names = input_schema.column_names
        LOGGER.warn(col_names)

        # get data type of the target column ==> double or int
        input_aggregation_column_ktype = (
            input_schema[:]
            .delegate._columns[col_names.index(self.aggreg_params.aggregation_column)]
            .ktype
        )

        LOGGER.warn(type(input_aggregation_column_ktype))

        # get knime data type of the selected timestamp column
        input_timestamp_ktype = (
            input_schema[:]
            .delegate._columns[col_names.index(self.aggreg_params.datetime_col)]
            .ktype
        )
        # TODO: Get rid of the LOGGER statements
        LOGGER.warning(f"{input_timestamp_ktype=}")

        # get the value factory string of the respective datetime type;
        # eg. ZonedDateTimeValueFactory2, LocalDateTimeValueFactory, LocalDateValueFactory and LocalTimeValueFactory
        time_stamp_logical_type = json.loads(input_timestamp_ktype.logical_type).get(
            "value_factory_class"
        )

        # TODO: Get rid of the LOGGER statements
        LOGGER.warning(time_stamp_logical_type)

        # if aggregation column is double, the final output of aggregation column after aggregation will always be double
        if input_aggregation_column_ktype == knext.double():
            # TODO: Get rid of the LOGGER statements
            LOGGER.warning("we are in double")
            # if input column is of type TIME, then return the following specs
            if time_stamp_logical_type == kutil.LOCAL_TIME_VALUE:

                # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.HOUR.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MINUTE.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.SECOND.name.lower(),
                ]:
                    if (  # if "count" method is encountered, then the output type of target will be long
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):

                        output_schema = self.__time_schema(True)

                    else:
                        output_schema = self.__time_schema(False)

                    return output_schema
                else:
                    raise ValueError(
                        f"Incompatible time granularity, {self.aggreg_params.time_granularity}, selected."
                    )

            elif time_stamp_logical_type == kutil.LOCAL_DATE_VALUE:
                # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
                ]:
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__year_schema_three_cols(True)
                    else:
                        output_schema = self.__year_schema_three_cols(False)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.DAY.name.lower()
                ):
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__date_schema(True)
                    else:
                        output_schema = self.__date_schema(False)

                    return output_schema

                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.YEAR.name.lower()
                ):
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__year_schema_two_cols(True)
                    else:
                        output_schema = self.__year_schema_two_cols(False)

                    return output_schema
                else:
                    raise ValueError(
                        f"Incompatible time granularity, {self.aggreg_params.time_granularity}, selected."
                    )

            elif time_stamp_logical_type == kutil.LOCAL_DATE_TIME_VALUE:
                #         # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
                ]:
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__year_schema_three_cols(True)
                    else:
                        output_schema = self.__year_schema_three_cols(False)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.DAY.name.lower()
                ):
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__date_schema(True)
                    else:
                        output_schema = self.__date_schema(False)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.YEAR.name.lower()
                ):
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__year_schema_two_cols(True)
                    else:
                        output_schema = self.__year_schema_two_cols(False)

                    return output_schema
                elif self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.HOUR.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MINUTE.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.SECOND.name.lower(),
                ]:
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__datetime_schema(True)
                    else:
                        output_schema = self.__datetime_schema(False)

                    return output_schema

            elif time_stamp_logical_type == kutil.ZONED_DATE_TIME_ZONE_VALUE:



            
                # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
                ]:
                    if (
                        # only count aggregation method from Pandas returns a Long or int64 data type.
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__year_schema_three_cols(True)
                    else:
                        output_schema = self.__year_schema_three_cols(False)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.DAY.name.lower()
                ):
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__date_schema(True)
                    else:
                        output_schema = self.__date_schema(False)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.YEAR.name.lower()
                ):
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__year_schema_two_cols(True)
                    else:
                        output_schema = self.__year_schema_two_cols(False)

                    return output_schema

                elif self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.HOUR.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MINUTE.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.SECOND.name.lower(),
                ]:
                    if (
                        self.aggreg_params.aggregation_methods.lower()
                        == self.aggreg_params.AggregationMethods.COUNT.name.lower()
                    ):
                        output_schema = self.__zoned_datetime_schema(True)
                    else:
                        output_schema = self.__zoned_datetime_schema(False)

                    return output_schema

        elif input_aggregation_column_ktype in [
            knext.int32(),
            knext.int64(),
        ]:  # knext.int64() has been pulled out since now this returns all target aggregations except for the method "Mean" and "Variance" of type long/int64
            # TODO: Get rid of the LOGGER statements
            LOGGER.warning("we are in integer/long")
            # if input column is of type TIME, then return the following specs
            if time_stamp_logical_type == kutil.LOCAL_TIME_VALUE:

                # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.HOUR.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MINUTE.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.SECOND.name.lower(),
                ]:
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:

                        output_schema = self.__time_schema(False)

                    else:
                        output_schema = self.__time_schema(True)

                    return output_schema
                else:
                    raise ValueError(
                        f"Incompatible time granularity, {self.aggreg_params.time_granularity}, selected."
                    )

            elif time_stamp_logical_type == kutil.LOCAL_DATE_VALUE:
                # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
                ]:
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__year_schema_three_cols(False)
                    else:
                        output_schema = self.__year_schema_three_cols(True)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.DAY.name.lower()
                ):
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__date_schema(False)
                    else:
                        output_schema = self.__date_schema(True)

                    return output_schema

                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.YEAR.name.lower()
                ):
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__year_schema_two_cols(False)
                    else:
                        output_schema = self.__year_schema_two_cols(True)

                    return output_schema
                else:
                    raise ValueError(
                        f"Incompatible time granularity, {self.aggreg_params.time_granularity}, selected."
                    )

            elif time_stamp_logical_type == kutil.LOCAL_DATE_TIME_VALUE:
                #         # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
                ]:
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__year_schema_three_cols(False)
                    else:
                        output_schema = self.__year_schema_three_cols(True)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.DAY.name.lower()
                ):
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__date_schema(False)
                    else:
                        output_schema = self.__date_schema(True)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.YEAR.name.lower()
                ):
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__year_schema_two_cols(False)
                    else:
                        output_schema = self.__year_schema_two_cols(True)

                    return output_schema
                elif self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.HOUR.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MINUTE.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.SECOND.name.lower(),
                ]:
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__datetime_schema(False)
                    else:
                        output_schema = self.__datetime_schema(True)

                    return output_schema

            elif time_stamp_logical_type == kutil.ZONED_DATE_TIME_ZONE_VALUE:
                # write logic here wrt to expected output after execution
                if self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
                ]:
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__year_schema_three_cols(False)
                        # output_schema = self.__create_schema(target_column_is_of_type_double=False, )
                    else:
                        output_schema = self.__year_schema_three_cols(True)
                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.DAY.name.lower()
                ):
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__date_schema(False)
                    else:
                        output_schema = self.__date_schema(True)

                    return output_schema
                elif (
                    self.aggreg_params.time_granularity.lower()
                    == self.aggreg_params.TimeGranularityOpts.YEAR.name.lower()
                ):
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__year_schema_two_cols(False)
                    else:
                        output_schema = self.__year_schema_two_cols(True)

                    return output_schema

                elif self.aggreg_params.time_granularity.lower() in [
                    self.aggreg_params.TimeGranularityOpts.HOUR.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.MINUTE.name.lower(),
                    self.aggreg_params.TimeGranularityOpts.SECOND.name.lower(),
                ]:
                    if self.aggreg_params.aggregation_methods.lower() in [
                        self.aggreg_params.AggregationMethods.MEAN.name.lower(),
                        self.aggreg_params.AggregationMethods.VARIANCE.name.lower(),
                    ]:
                        output_schema = self.__zoned_datetime_schema(False)
                    else:
                        output_schema = self.__zoned_datetime_schema(True)

                    return output_schema
        LOGGER.warn(f"{output_schema=}")
        return output_schema

    # def __year_schema_three_cols(self, switch_agg=False):

    #     if switch_agg:
    #         return knext.Schema(
    #             [
    #                 knext.int32(),
    #                 knext.int32(),
    #                 knext.int32(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.time_granularity.lower(),
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )
    #     else:
    #         return knext.Schema(
    #             [
    #                 knext.int32(),
    #                 knext.int32(),
    #                 knext.double(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.time_granularity.lower(),
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )

    # def __year_schema_two_cols(self, switch_agg=False):

    #     if switch_agg:
    #         return knext.Schema(
    #             [
    #                 knext.int32(),
    #                 knext.int32(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )
    #     else:
    #         return knext.Schema(
    #             [
    #                 knext.int32(),
    #                 knext.double(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )

    # def __date_schema(self, switch_agg=False):

    #     if switch_agg:
    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=True, time=False, timezone=False),
    #                 knext.int32(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )
    #     else:

    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=True, time=False, timezone=False),
    #                 knext.double(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )

    # def __datetime_schema(self, switch_agg=False):

    #     if switch_agg:
    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=True, time=True, timezone=False),
    #                 knext.int32(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )
    #     else:

    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=True, time=True, timezone=False),
    #                 knext.double(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )

    class AggregationCategory(Enum):
        YEAR = 1
        DAY_OR_SHORTER = 2
        WEEK_OR_LONGER = 3

    # "switch_agg": whether the output column 'target' is of type double or int - target_column_is_of_type_double
    def __create_schema(
        self,
        target_column_is_of_type_double: bool,
        contains_timezone: bool,
        contains_time: bool,
        contains_date: bool,
        aggregation_category: AggregationCategory,
    ):

        # specify first column
        if aggregation_category is not self.AggregationCategory.DAY_OR_SHORTER:
            data_types = [knext.int32()]  # TODO doublecheck
        else:
            data_types = [
                knext.datetime(
                    date=contains_date, time=contains_time, timezone=contains_timezone
                )
            ]

        column_names = [self.aggreg_params.datetime_col]

        # specify optional second column
        if self.AggregationCategory.WEEK_OR_LONGER:
            data_types.append(knext.int32())
            column_names.append(self.aggreg_params.time_granularity.lower())

        # specify third column
        data_types.append(
            knext.double() if target_column_is_of_type_double else knext.int64()
        )
        column_names.append(self.aggreg_params.aggregation_column)

        return knext.Schema(data_types, column_names)

    # switch_agg is the boolean flag to enable int or double data type in the aggregated data type. If false, then type will be double and int otherwise.
    # Pandas upon aggregation on "COUNT" returns the aggregated column in int64/Long format. if false, then ktype will be int32 (means aggregation methos is not count) else int64.
    # def __time_schema(self, switch_agg=False):

    #     if switch_agg:
    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=False, time=True, timezone=False),
    #                 knext.int32(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )
    #     else:

    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=False, time=True, timezone=False),
    #                 knext.double(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )

    # def __zoned_datetime_schema(self, switch_agg=False):

    #     if switch_agg:
    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=True, time=True, timezone=True),
    #                 knext.int64(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )
    #     else:

    #         return knext.Schema(
    #             [
    #                 knext.datetime(date=True, time=True, timezone=True),
    #                 knext.double(),
    #             ],
    #             [
    #                 self.aggreg_params.datetime_col,
    #                 self.aggreg_params.aggregation_column,
    #             ],
    #         )

    def execute(self, exec_context: knext.ExecutionContext, input_1: knext.Schema):
        df = input_1.to_pandas()

        date_time_col_orig = df[self.aggreg_params.datetime_col]
        agg_col = df[self.aggreg_params.aggregation_column]

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
        selected_time_granularity = self.aggreg_params.time_granularity.lower()

        # this variable is assigned the aggregation method selected by the user
        selected_aggreg_method = self.aggreg_params.aggregation_methods.lower()

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

        # down cast aggregation to int32
        df_grouped[self.aggreg_params.aggregation_column] = df_grouped[
            self.aggreg_params.aggregation_column
        ].astype(np.int32)

        exec_context.set_progress(0.8)
        if selected_time_granularity not in (
            self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
            self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
            self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
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

        # TODO: mention in commit message about the reconverting to datetime and then just extracting the time values
        if kn_date_time_type == kutil.DEF_TIME_LABEL:
            date_col = pd.to_datetime(
                df[self.aggreg_params.datetime_col], format=kutil.TIME_FORMAT
            ).dt.strftime("%H:%M:%S")
        else:
            date_col = df[self.aggreg_params.datetime_col].astype("datetime64[ns]")

        # check if granularity level is
        if time_gran in (
            self.aggreg_params.TimeGranularityOpts.YEAR.name.lower(),
            self.aggreg_params.TimeGranularityOpts.QUARTER.name.lower(),
            self.aggreg_params.TimeGranularityOpts.MONTH.name.lower(),
            self.aggreg_params.TimeGranularityOpts.WEEK.name.lower(),
        ):
            # return year only
            date_col = date_col.dt.year
            df[self.aggreg_params.datetime_col] = date_col
            df[self.aggreg_params.datetime_col] = df[
                self.aggreg_params.datetime_col
            ].astype(np.int32)

        # set input timestamp to date_col
        elif time_gran == self.aggreg_params.TimeGranularityOpts.DAY.name.lower():

            date_col = date_col.dt.date

            df[self.aggreg_params.datetime_col] = date_col

        # round datetime to nearest hour
        elif time_gran == self.aggreg_params.TimeGranularityOpts.HOUR.name.lower():
            df[self.aggreg_params.datetime_col] = self.__floor_time(
                kn_date_time_type, "H", date_col
            )

        # round datetime to nearest minute
        elif time_gran == self.aggreg_params.TimeGranularityOpts.MINUTE.name.lower():
            df[self.aggreg_params.datetime_col] = self.__floor_time(
                kn_date_time_type, "min", date_col
            )

        # round datetime to nearest second. This option is feasble if timestamp contains milliseconds/microseconds/nanoseconds.
        elif time_gran == self.aggreg_params.TimeGranularityOpts.SECOND.name.lower():
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
