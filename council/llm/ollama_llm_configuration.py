from __future__ import annotations

from typing import Any, Dict, Final, Literal, Mapping, Optional, Tuple, Type, Union

from council.utils import Parameter, greater_than_validator, read_env_str

from . import LLMConfigSpec, LLMConfigurationBase

_env_var_prefix: Final[str] = "OLLAMA_"


def _tv(x: float) -> None:
    """
    # TODO: refactor
    Temperature and Top_p Validators
    Sampling temperature to use, between 0. and 1.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("must be in the range [0.0..1.0]")


class OllamaLLMConfiguration(LLMConfigurationBase):
    def __init__(
        self, model: str, keep_alive: Optional[Union[float, str]] = None, format: Literal["", "json"] = ""
    ) -> None:
        """
        Initialize a new instance

        Args:
            model (str): model name to use from https://ollama.com/library

        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model)
        self._keep_alive = keep_alive
        self._format = format

        # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        self._mirostat = Parameter.int(name="mirostat", required=False)
        self._mirostat_eta = Parameter.float(name="mirostat_eta", required=False)
        self._mirostat_tau = Parameter.float(name="mirostat_tau", required=False)
        self._num_ctx = Parameter.int(name="num_ctx", required=False, validator=greater_than_validator(0))
        self._repeat_last_n = Parameter.int(name="repeat_last_n", required=False)
        self._repeat_penalty = Parameter.float(name="repeat_penalty", required=False)
        self._temperature = Parameter.float(name="temperature", required=False, default=0.0, validator=_tv)
        self._seed = Parameter.int(name="seed", required=False)
        self._stop = Parameter.string(name="stop", required=False)
        self._tfs_z = Parameter.float(name="tfs_z", required=False)
        self._num_predict = Parameter.int(name="num_predict", required=False)
        self._top_p = Parameter.float(name="top_p", required=False, validator=_tv)
        self._top_k = Parameter.int(name="top_k", required=False, validator=greater_than_validator(0))
        self._min_p = Parameter.float(name="min_p", required=False)

    def model_name(self) -> str:
        return self._model.unwrap()

    @property
    def model(self) -> Parameter[str]:
        """Ollama model."""
        return self._model

    @property
    def keep_alive(self) -> Optional[Union[float, str]]:
        """
        Number of seconds / duration string to keep model in memory.
        See https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately
        """
        return self._keep_alive

    @property
    def format(self) -> Literal["", "json"]:
        """The format to return a response in."""
        return self._format

    @property
    def mirostat(self) -> Parameter[int]:
        """Enable Mirostat sampling for controlling perplexity."""
        return self._mirostat

    @property
    def mirostat_eta(self) -> Parameter[float]:
        """Learning rate for Mirostat sampling."""
        return self._mirostat_eta

    @property
    def mirostat_tau(self) -> Parameter[float]:
        """Controls balance between coherence and diversity."""
        return self._mirostat_tau

    @property
    def num_ctx(self) -> Parameter[int]:
        """Context window size."""
        return self._num_ctx

    @property
    def repeat_last_n(self) -> Parameter[int]:
        """Look back size for repetition prevention."""
        return self._repeat_last_n

    @property
    def repeat_penalty(self) -> Parameter[float]:
        """Penalty for repetition."""
        return self._repeat_penalty

    @property
    def temperature(self) -> Parameter[float]:
        """The temperature of the model."""
        return self._temperature

    @property
    def seed(self) -> Parameter[int]:
        """Random seed."""
        return self._seed

    @property
    def stop(self) -> Parameter[str]:
        """Stop sequence."""
        return self._stop

    @property
    def tfs_z(self) -> Parameter[float]:
        """Tail free sampling parameter."""
        return self._tfs_z

    @property
    def num_predict(self) -> Parameter[int]:
        """Maximum number of tokens to predict."""
        return self._num_predict

    @property
    def top_k(self) -> Parameter[int]:
        """
        Only sample from the top K options for each subsequent token.
        Used to remove "long tail" low probability responses.
        """
        return self._top_k

    @property
    def top_p(self) -> Parameter[float]:
        """
        Use nucleus sampling.
        In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in
        decreasing probability order and cut it off once it reaches a particular probability specified by top_p.
        """
        return self._top_p

    @property
    def min_p(self) -> Parameter[float]:
        """Minimum probability for token consideration"""
        return self._min_p

    def params_to_options(self) -> Dict[str, Any]:
        """Convert parameters to options dict"""
        return {
            "mirostat": self.mirostat.value,
            "mirostat_eta": self.mirostat_eta.value,
            "mirostat_tau": self.mirostat_tau.value,
            "num_ctx": self.num_ctx.value,
            "repeat_last_n": self.repeat_last_n.value,
            "repeat_penalty": self.repeat_penalty.value,
            "temperature": self.temperature.value,
            "seed": self.seed.value,
            "stop": self.stop.value,
            "tfs_z": self.tfs_z.value,
            "num_predict": self.num_predict.value,
            "top_k": self.top_k.value,
            "top_p": self.top_p.value,
            "min_p": self.min_p.value,
        }

    @staticmethod
    def from_env() -> OllamaLLMConfiguration:
        model = read_env_str(_env_var_prefix + "LLM_MODEL").unwrap()
        keep_alive_env_str = read_env_str(_env_var_prefix + "KEEP_ALIVE", required=False)
        keep_alive = keep_alive_env_str.unwrap() if keep_alive_env_str.is_some() else None

        config = OllamaLLMConfiguration(model=model, keep_alive=keep_alive)
        return config

    @staticmethod
    def from_spec(spec: LLMConfigSpec) -> OllamaLLMConfiguration:
        model = spec.provider.must_get_value("model")
        config = OllamaLLMConfiguration(model=str(model))

        value = spec.provider.get_value("keep_alive")
        if value is not None:
            config._keep_alive = value

        value = spec.provider.get_value("format")
        if value is not None:
            config._format = value

        if spec.parameters is not None:
            param_mapping: Mapping[str, Tuple[Parameter, Type]] = {
                "mirostat": (config.mirostat, int),
                "mirostatEta": (config.mirostat_eta, float),
                "mirostatTau": (config.mirostat_tau, float),
                "numCtx": (config.num_ctx, int),
                "repeatLastN": (config.repeat_last_n, int),
                "repeatPenalty": (config.repeat_penalty, float),
                "temperature": (config.temperature, float),
                "seed": (config.seed, int),
                "stop": (config.stop, list),
                "tfsZ": (config.tfs_z, float),
                "numPredict": (config.num_predict, int),
                "topK": (config.top_k, int),
                "topP": (config.top_p, float),
                "minP": (config.min_p, float),
            }

            for key, (param, type_conv) in param_mapping.items():
                value = spec.parameters.get(key)
                if value is not None:
                    param.set(type_conv(value))

        return config
