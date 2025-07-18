"""
Several utility functions are reused from Harvard's spatial data lab repository for Geospatial Analytics Extension.
https://github.com/spatial-data-lab/knime-geospatial-extension/blob/main/knime_extension/src/util/knime_utils.py
"""

import knime.extension as knext
import pandas as pd
from typing import Callable
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

category = knext.category(
    path="/labs",
    level_id="ts",
    name="KNIME Timeseries Analysis Extension",
    description="Python Nodes for Time Series Analysis",
    icon="icons/Time_Series_Analysis.png",
)

############################################
# Timestamp column selection helper
############################################

# Strings of IDs of date/time value factories
ZONED_DATE_TIME_ZONE_VALUE = "org.knime.core.data.v2.time.ZonedDateTimeValueFactory2"
LOCAL_TIME_VALUE = "org.knime.core.data.v2.time.LocalTimeValueFactory"
LOCAL_DATE_VALUE = "org.knime.core.data.v2.time.LocalDateValueFactory"
LOCAL_DATE_TIME_VALUE = "org.knime.core.data.v2.time.LocalDateTimeValueFactory"


DEF_ZONED_DATE_LABEL = "ZonedDateTimeValueFactory2"
DEF_DATE_LABEL = "LocalDateValueFactory"
DEF_TIME_LABEL = "LocalTimeValueFactory"
DEF_DATE_TIME_LABEL = "LocalDateTimeValueFactory"

# Timestamp formats
ZONED_DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S%z"
DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"


############################################
# Categories
############################################
BASE_CATEGORY_PATH = "/labs/ts"

category_processsing = knext.category(
    path=BASE_CATEGORY_PATH,
    level_id="proc",
    name="Preprocessing",
    description="Nodes for pre-processing timestamp data.",
    icon="icons/Pre-processing.png",
)

category_models = knext.category(
    path=BASE_CATEGORY_PATH,
    level_id="models",
    name="Models",
    description="Nodes for modelling time series",
    icon="icons/Models.png",
)

category_analytics = knext.category(
    path=BASE_CATEGORY_PATH,
    level_id="analysis",
    name="Analysis",
    description="Nodes for analysis for time series application",
    icon="icons/Analytics.png",
)


def is_numeric(column: knext.Column) -> bool:
    """
    Checks if column is numeric e.g. int, long or double.
    @return: True if Column is numeric
    """
    return (
        column.ktype == knext.double()
        or column.ktype == knext.int32()
        or column.ktype == knext.int64()
    )


def is_zoned_datetime(column: knext.Column) -> bool:
    """
    Checks if date&time column contains has the timezone or not.
    @return: True if selected date&time column has time zone
    """
    return __is_type_x(column, ZONED_DATE_TIME_ZONE_VALUE)


def is_datetime(column: knext.Column) -> bool:
    """
    Checks if a column is of type Date&Time.
    @return: True if selected column is of type date&time
    """
    return __is_type_x(column, LOCAL_DATE_TIME_VALUE)


def is_time(column: knext.Column) -> bool:
    """
    Checks if a column is of type Time only.
    @return: True if selected column has only time.
    """
    return __is_type_x(column, LOCAL_TIME_VALUE)


def is_date(column: knext.Column) -> bool:
    """
    Checks if a column is of type date only.
    @return: True if selected column has date only.
    """
    return __is_type_x(column, LOCAL_DATE_VALUE)


def boolean_or(*functions):
    """
    Return True if any of the given functions returns True
    @return: True if any of the functions returns True
    """

    def new_function(*args, **kwargs):
        return any(f(*args, **kwargs) for f in functions)

    return new_function


def is_type_timestamp(column: knext.Column):
    """
    This function checks on all the supported timestamp columns supported in KNIME.
    Note that legacy date&time types are not supported.
    @return: True if date&time column is compatible with the respective logical types supported in KNIME.
    """

    return boolean_or(is_time, is_date, is_datetime, is_zoned_datetime)(column)


def __is_type_x(column: knext.Column, type: str) -> bool:
    """
    Checks if column contains the given type whereas type can be :
    DateTime, Date, Time, ZonedDateTime
    @return: True if column type is of type timestamp
    """

    return (
        isinstance(column.ktype, knext.LogicalType)
        and type in column.ktype.logical_type
    )


############################################
# Date&Time helper methods
############################################


def convert_timestamp(value):
    """
    Converts a value to a pandas Timestamp.
    Raises ValueError if conversion fails.
    """
    try:
        return pd.Timestamp(value)
    except Exception as e:
        raise ValueError(f"Failed to convert value '{value}' to pandas Timestamp: {e}")

def safe_to_datetime(series, format=None):
    """
    Converts a pandas Series to datetime, raising ValueError on failure.
    """
    try:
        if format:
            return pd.to_datetime(series, format=format)
        else:
            return pd.to_datetime(series)
    except Exception as e:
        raise ValueError(f"Failed to convert series to datetime with format '{format}': {e}")
    
def extract_zone(value):
    """
    This function extracts the time zone from each timestamp value in the pandas timmestamp column.
    @return: timezone of Timestamp
    """
    return value.tz


def localize_timezone(value: pd.Timestamp, zone) -> pd.Timestamp:
    """
    This function updates the Pandas Timestamp value with the time zone. If "None" is passed timezone will be removed from
    column returning a value with no timezone.
    @return: assigns timezone to timestamp.

    """
    return value.tz_localize(zone)


def time_granularity_list() -> list:
    """
    This function returns list of possible time fields relevant to only time type values.
    @return: list of item fields in Time
    """
    return [
        "Hour",
        "Minute",
        "Second",
        # not supported yet
        "Millisecond",
        "Microsecond",
    ]


def cast_to_related_type(
    value_type: str, column: pd.Series
) -> tuple[pd.Series, str] | tuple[pd.Series, str, pd.Series]:
    """
    Converts a KNIME timestamp column to a Pandas native timestamp format.

    Takes a KNIME timestamp column and converts it to the appropriate Pandas timestamp format based on the
    KNIME factory type. Handles zoned datetimes, dates only, times only, and datetime values.

    Args:
        value_type: The KNIME timestamp factory type string (e.g. "ZonedDateTimeValueFactory2")
        column: The pandas Series containing the timestamp values to convert

    Returns:
        For zoned datetimes:
            A tuple of (converted datetime Series, factory type string, timezone offset Series)
        For other types:
            A tuple of (converted datetime Series, factory type string)

    Raises:
        ValueError: If an invalid timestamp is provided
    """

    conversion_mappings = {
        DEF_ZONED_DATE_LABEL: {
            "preprocess": lambda col: col.apply(convert_timestamp),
            "extract_zone": lambda col: col.apply(extract_zone),
            "localize": lambda col: col.apply(localize_timezone, zone=None),
            "convert": lambda col: safe_to_datetime(col, format=ZONED_DATE_TIME_FORMAT),
            "return_zone": True,
        },
        DEF_DATE_LABEL: {
            "convert": lambda col: safe_to_datetime(col, format=DATE_FORMAT),
            "return_zone": False,
        },
        DEF_TIME_LABEL: {
            "convert": lambda col: safe_to_datetime(col, format=TIME_FORMAT).dt.time,
            "return_zone": False,
        },
        DEF_DATE_TIME_LABEL: {
            "convert": lambda col: safe_to_datetime(col, format=DATE_TIME_FORMAT),
            "return_zone": False,
        },
    }

    if value_type not in conversion_mappings:
        raise ValueError(f"Unsupported timestamp type: {value_type}")

    mapping = conversion_mappings[value_type]

    if "preprocess" in mapping:
        column = mapping["preprocess"](column)

    zone_offset = None
    if mapping.get("return_zone", False):
        zone_offset = mapping["extract_zone"](column)
        column = mapping["localize"](column)

    result = mapping["convert"](column)

    if mapping.get("return_zone", False):
        return result, value_type, zone_offset
    else:
        return result, value_type


def extract_time_fields(
    date_time_col: pd.Series, date_time_format: str, series_name: str
) -> pd.DataFrame:
    """
    This function exracts the timestamp fields in seperate columns.
    @return: Pandas dataframe with a timestamp column and relevant date&time fields.
    """

    field_mappings = {
        DEF_ZONED_DATE_LABEL: {
            "format": ZONED_DATE_TIME_FORMAT,
            "fields": [
                "year",
                "quarter",
                "month",
                "week",
                "day",
                "hour",
                "minute",
                "second",
            ],
            "special_fields": {"zone": lambda col: str(col.dt.tz)},
            "final_transform": None,
        },
        DEF_DATE_LABEL: {
            "format": DATE_FORMAT,
            "fields": ["year", "quarter", "month", "week", "day"],
            "special_fields": {},
            "final_transform": lambda col: col.dt.date,
        },
        DEF_TIME_LABEL: {
            "format": TIME_FORMAT,
            "fields": ["hour", "minute", "second"],
            "special_fields": {},
            "final_transform": lambda col: col.dt.time,
        },
        DEF_DATE_TIME_LABEL: {
            "format": DATE_TIME_FORMAT,
            "fields": [
                "year",
                "quarter",
                "month",
                "week",
                "day",
                "hour",
                "minute",
                "second",
            ],
            "special_fields": {},
            "final_transform": None,
        },
    }

    if date_time_format not in field_mappings:
        raise ValueError(f"Unsupported date_time_format: {date_time_format}")

    mapping = field_mappings[date_time_format]

    #following line can be redundant, in case the date&time aggregator and date&time aligner 
    # can function with all supported datetime types, conversion to 'to_datetime()' from 
    # and already converted series would not be needed.
    df = pd.to_datetime(date_time_col, format=mapping["format"]).to_frame(
        name=series_name
    )

    for field in mapping["fields"]:
        if field == "week":
            df[field] = df[series_name].dt.isocalendar().week.astype(np.int32)
        else:
            df[field] = getattr(df[series_name].dt, field)

    for field_name, field_func in mapping["special_fields"].items():
        df[field_name] = field_func(df[series_name])

    if mapping["final_transform"]:
        df[series_name] = mapping["final_transform"](df[series_name])

    cols_cap = [series_name] + [
        col.capitalize() for col in df.columns if col != series_name
    ]
    df.columns = cols_cap

    return df


def get_type_timestamp(value_type):
    """
    This function parses the complete value of KNIME's date&time factory type and returns the actual name of the factory data type.
    """
    typs = [
        ZONED_DATE_TIME_ZONE_VALUE,
        LOCAL_TIME_VALUE,
        LOCAL_DATE_VALUE,
        LOCAL_DATE_TIME_VALUE,
    ]

    for typ in typs:
        if str(value_type).__contains__(typ):
            type_detected = typ.split(".")
            return type_detected[len(type_detected) - 1]


############################################
# General Helper Class
############################################


def column_exists_or_preset(
    context: knext.ConfigurationContext,
    column: str,
    schema: knext.Schema,
    func: Callable[[knext.Column], bool] = None,
    none_msg: str = "No compatible column found in input table",
) -> str:
    """
    Checks that the given column is not None and exists in the given schema. If none is selected it returns the
    first column that is compatible with the provided function. If none is compatible it throws an exception.
    """
    if column is None:
        for c in schema:
            if func(c):
                context.set_warning(f"Preset column to: {c.name}")
                return c.name
        raise knext.InvalidParametersError(none_msg)
    __check_col_and_type(column, schema, func)
    return column


def __check_col_and_type(
    column: str,
    schema: knext.Schema,
    check_type: Callable[[knext.Column], bool] = None,
) -> None:
    """
    Checks that the given column exists in the given schema and that it matches the given type_check function.
    """
    # Check that the column exists in the schema and that it has a compatible type
    try:
        existing_column = schema[column]
        if check_type is not None and not check_type(existing_column):
            raise knext.InvalidParametersError(
                f"Column '{str(column)}' has incompatible data type"
            )
    except IndexError:
        raise knext.InvalidParametersError(
            f"Column '{str(column)}' not available in input table"
        )


############################################
# Generic pandas dataframe/series helper function
############################################


def check_missing_values(column: pd.Series) -> bool:
    """
    This function checks for missing values in the Pandas Series.
    @return: True if missing values exist in column
    """
    return column.hasnans


def count_missing_values(column: pd.Series) -> int:
    """
    This function counts the number of missing values in the Pandas Series.
    @return: sum of boolean 1s if missing value exists.
    """
    return column.isnull().sum()


def number_of_rows(df: pd.Series) -> int:
    """
    This function returns the number of rows in the dataframe.
    @return: numerical value, denoting length of Pandas Series.
    """
    return len(df.index)


def count_negative_values(column: pd.Series) -> int:
    total_neg = (column <= 0).sum()

    return total_neg



