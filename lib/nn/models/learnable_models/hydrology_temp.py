from typing import List, Literal, Union

from lib.nn.models.learnable_models.time_then_graph_isotropic import TimeThenGraphIsoModel


class HydrologyTempModel(TimeThenGraphIsoModel):
    """Temperature forecasting model inspired by the repo’s Time-Then-Graph
    isotropic design, with defaults suited for scalar targets (e.g., temperature)
    and rich exogenous inputs (extended NWP + topo).

    It uses a GRU temporal encoder followed by stacked graph message passing
    layers, plus a sampling readout for probabilistic predictions compatible
    with the existing training pipeline.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 64,
        emb_size: int = 16,
        add_embedding_before: Union[str, List[str]] = ("encoding", "message_passing"),
        use_local_weights: Union[str, List[str]] | None = None,
        time_layers: int = 2,
        graph_layers: int = 2,
        root_weight: bool = True,
        norm: str = "sym",
        add_backward: bool = False,
        cached: bool = False,
        activation: str = "elu",
        noise_mode: Literal["lin", "multi", "add", "none"] = "lin",
        time_skip_connect: bool = True,
    ):
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            emb_size=emb_size,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            time_layers=time_layers,
            graph_layers=graph_layers,
            root_weight=root_weight,
            norm=norm,
            add_backward=add_backward,
            cached=cached,
            activation=activation,
            noise_mode=noise_mode,
            time_skip_connect=time_skip_connect,
        )

