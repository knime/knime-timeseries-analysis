import knime.extension as knext
import logging

LOGGER = logging.getLogger(__name__)


category = knext.category(
    path="/labs",
    level_id="ts",
    name="KNIME Timeseries Analysis Extension",
    description="Python Nodes for Time Series Analysis",
    icon="icons/Time_Series_Analysis.png",
)

from nodes.models import (  # noqa: E402
    sarimanode,  # noqa: F401
    sarimaxnode,  # noqa: F401
    sarima_apply_node,  # noqa: F401
    sarimax_apply_node,  # noqa: F401
)  # NOSONAR
from nodes.preprocessing import (  # noqa: E402
    aggreg_gran_node,  # noqa: F401
    differencing_node,  # noqa: F401
    timestamp_alignment_node,  # noqa: F401
)  # NOSONAR
from nodes.analysis import autocorrnode, residuals_analyzer_node  # noqa: E402, F401
