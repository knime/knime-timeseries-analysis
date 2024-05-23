import logging
import knime.extension as knext
from util import utils as kutil
from ..configs.models.sarimax_apply import SarimaxForecasterParams
import numpy as np
import pickle

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="SARIMAX Predictor (Labs)",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/models/SARIMAX_Forecaster-Apply.png",
    category=kutil.category_models,
    id="sarimax_apply",
)
@knext.input_binary(
    name="Input Data", description="Trained SARIMAX model.", id="sarimax.model"
)
@knext.input_table(name="Exogenous Input", description="Link to exogenous variable")
@knext.output_table(
    name="Forecast", description="Forecasted values and their standard errors"
)
class SXForecaster:
    """
    This node  generates forecasts with a (S)ARIMAX Model.

    Based on a trained SARIMAX model given at the model input port of this node, the forecast values are computed. This apply node can also be used to update exogenous variable data for forecasting.
    """

    sarimax_params = SarimaxForecasterParams()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1,  # NOSONAR
        input_schema_2,
    ):
        # set exog input for forecasting
        self.sarimax_params.predictor_params.exog_column_forecasts = (
            kutil.column_exists_or_preset(
                configure_context,
                self.sarimax_params.predictor_params.exog_column_forecasts,
                input_schema_2,
                kutil.is_numeric,
            )
        )
        return knext.Column(knext.double(), "Forecasts")

    def execute(self, exec_context: knext.ExecutionContext, input_1, input_2):
        model_fit = pickle.loads(input_1)
        exog_df = input_2.to_pandas()
        exog_forecasts_col = exog_df[
            self.sarimax_params.predictor_params.exog_column_forecasts
        ]
        exec_context.set_progress(0.2)

        self.__validate(exog_forecasts_col)

        # make out-of-sample forecasts
        forecasts = model_fit.forecast(
            steps=self.sarimax_params.predictor_params.number_of_forecasts,
            exog=exog_forecasts_col,
        ).to_frame(name="Forecasts")

        exec_context.set_progress(0.7)

        # reverse log transformation for forecasts
        if self.sarimax_params.predictor_params.natural_log:
            forecasts = np.exp(forecasts)

        exec_context.set_progress(0.9)

        return knext.Table.from_pandas(forecasts)

    # function to perform validation on dataframe within execution context
    def __validate(self, exog_forecast_col):
        if kutil.check_missing_values(exog_forecast_col):
            missing_count_exog_fore = kutil.count_missing_values(exog_forecast_col)
            raise knext.InvalidParametersError(
                f"""There are {missing_count_exog_fore} missing values in the exogenous column selected for forecasting."""
            )

        if (
            kutil.number_of_rows(exog_forecast_col)
            != self.sarimax_params.predictor_params.number_of_forecasts
        ):
            raise knext.InvalidParametersError(
                "The number of forecasts should be equal to the length of the exogenous input for forecasts."
            )
