import logging
import threading
from typing import Any, Dict, Iterator, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    generated_text: str
    generated_token_count: int
    input_token_count: int
    stop_reason: str


class WatsonXResponse(BaseModel):
    model_id: str
    results: List[GenerationResult]


class LangchainWatsonBam(BaseLLM):

    model_id: str
    """Type of model to use."""

    url: str
    """Url to Watson Machine Learning instance"""

    apikey: SecretStr
    """Apikey to Watson Machine Learning instance"""

    params: Optional[dict] = None
    """Model parameters to use during generate requests."""

    streaming: bool = False
    """ Whether to stream the results or not. """

    watsonx_model: Any

    max_workers: int = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "url": "WATSONX_URL",
            "apikey": "WATSONX_APIKEY",
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that credentials and python package exists in environment."""
        values["apikey"] = convert_to_secret_str(
            get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
        )
        values["url"] = get_from_dict_or_env(values, "url", "WATSONX_URL")

        watsonx_model = WatsonxBamClient(
            model_id=values["model_id"],
            apikey=values["apikey"].get_secret_value(),
            url=values["url"],
            max_workers=values["max_workers"],
        )
        values["watsonx_model"] = watsonx_model

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "params": self.params,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "IBM watsonx.ai"

    @staticmethod
    def _extract_token_usage(
        generations: Optional[List[GenerationResult]] = None,
    ) -> Dict[str, Any]:
        if generations is None:
            return {"generated_token_count": 0, "input_token_count": 0}

        input_token_count = 0
        generated_token_count = 0

        for generation in generations:
            input_token_count += generation.input_token_count
            generated_token_count += generation.generated_token_count

        return {
            "generated_token_count": generated_token_count,
            "input_token_count": input_token_count,
        }

    def _get_chat_params(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {**self.params} if self.params else {}
        if stop is not None:
            params["stop_sequences"] = stop
        return params

    def _create_llm_result(self, response: WatsonXResponse) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations: List[List[Generation]] = []
        for result in response.results:
            finish_reason = result.stop_reason
            gen = Generation(
                text=result.generated_text,
                generation_info={"finish_reason": finish_reason},
            )
            generations.append([gen])

        token_usage = self._extract_token_usage(response.results)

        llm_output = {
            "token_usage": token_usage,
            "model_id": response.model_id,
        }
        return LLMResult(generations=generations, llm_output=llm_output)

    def _stream_response_to_generation_chunk(
        self,
        stream_response: Dict[str, Any],
    ) -> GenerationChunk:
        """Convert a stream response to a generation chunk."""
        if not stream_response["results"]:
            return GenerationChunk(text="")
        return GenerationChunk(
            text=stream_response["results"][0]["generated_text"],
            generation_info=dict(
                finish_reason=stream_response["results"][0].get("stop_reason", None),
                llm_output={
                    "generated_token_count": stream_response["results"][0].get(
                        "generated_token_count", None
                    ),
                    "model_id": self.model_id,
                },
            ),
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the IBM watsonx.ai inference endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python

                response = watsonx_llm("What is a molecule")
        """
        result = self._generate(
            prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs
        )
        return result.generations[0][0].text

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call the IBM watsonx.ai inference endpoint which then generate the response.
        Args:
            prompts: List of strings (prompts) to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The full LLMResult output.
        Example:
            .. code-block:: python

                response = watsonx_llm.generate(["What is a molecule"])
        """
        params = self._get_chat_params(stop=stop)
        response: WatsonXResponse = self.watsonx_model.generate(
            prompts=prompts, params=params
        )
        return self._create_llm_result(response)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call the IBM watsonx.ai inference endpoint which then streams the response.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The iterator which yields generation chunks.
        Example:
            .. code-block:: python

                response = watsonx_llm.stream("What is a molecule")
                for chunk in response:
                    print(chunk, end='')
        """
        params = self._get_chat_params(stop=stop)
        for stream_resp in self.watsonx_model.generate_text_stream(
            prompt=prompt, raw_response=True, params=params
        ):
            chunk = self._stream_response_to_generation_chunk(stream_resp)
            yield chunk

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)


class WatsonxBamClient:
    def __init__(
        self, apikey: str, url: str, model_id: str, max_workers: Optional[int]
    ):
        self._headers = {
            "Authorization": f"Bearer {apikey}",
            "Content-Type": "application/json",
        }

        self._url = url
        self._model_id = model_id
        self._workers = max_workers

        if max_workers is not None:
            self._semaphore = threading.Semaphore(max_workers)

    def generate(self, prompts: list, params: dict) -> WatsonXResponse:

        generations = []
        for prompt in prompts:
            payload = {
                "parameters": params,
                "input": "[INST]" + prompt + "[/INST]",
                "model_id": self._model_id,
                "moderations": {},
            }

            if self._workers is not None:
                self._semaphore.acquire()

            resp = requests.post(self._url, headers=self._headers, json=payload)

            if self._workers is not None:
                self._semaphore.release()

            generations.extend(resp.json()["results"])

        return WatsonXResponse(results=generations, model_id=self._model_id)
