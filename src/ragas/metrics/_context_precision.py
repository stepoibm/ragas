from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)

CONTEXT_PRECISION = Prompt(
    name="context_precision",
    instruction="""Given the question, answer and context verify if the context was useful in arriving at the given answer. Give a json object with keys verdict and reason. Set the key "verdict" to "1" if useful and "0" if not. Set the key "reason" with an explanation for the verdict. Only give one json object.""",
    examples=[
        {
            "question": """What can you tell me about albert Albert Einstein?""",
            "context": """Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
            "answer": """Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895""",
            "verification": {
                "reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                "verdict": "1",
            },
        },
        {
            "question": """who won 2020 icc world cup?""",
            "context": """The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.""",
            "answer": """England""",
            "verification": {
                "reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                "verdict": "1",
            },
        },
        {
            "question": """What is the tallest mountain in the world?""",
            "context": """The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.""",
            "answer": """Mount Everest.""",
            "verification": {
                "reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                "verdict": "0",
            },
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="verification",
    output_type="json",
)


@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    evaluation_mode: EvaluationMode
    context_precision_prompt: Prompt
    """

    name: str = "context_precision"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    context_precision_prompt: Prompt = field(default_factory=lambda: CONTEXT_PRECISION)

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        answer = "ground_truth"
        if answer not in row.keys():
            logger.warning(
                "Using 'context_precision' without ground truth will be soon depreciated. Use 'context_utilization' instead"
            )
            answer = "answer"

        return row["question"], row["contexts"], row[answer]

    def _context_precision_prompt(self, row: t.Dict) -> t.List[PromptValue]:
        question, contexts, answer = self._get_row_attributes(row)
        return [
            self.context_precision_prompt.format(
                question=question, context=c, answer=answer
            )
            for c in contexts
        ]

    def _calculate_average_precision(self, json_responses: t.List[t.Dict]) -> float:
        score = np.nan
        data = [item if isinstance(item, dict) else {} for item in json_responses]

        print(data)

        verdict_list = [
            (
                int("1" == resp.get("verdict", "").strip())
                if resp.get("verdict")
                else np.nan
            )
            for resp in data
        ]
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        if np.isnan(score):
            logger.warning(
                f"Invalid response format. Expected a list of dictionaries with keys 'verdict'.\nResponses\n{json_responses}"
            )
        return score

    async def _ascore(
        self: t.Self,
        row: t.Dict,
        callbacks: Callbacks,
        is_async: bool,
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        human_prompts = self._context_precision_prompt(row)
        responses: t.List[str] = []
        for hp in human_prompts:
            result = await self.llm.generate(
                hp,
                n=1,
                callbacks=callbacks,
                is_async=is_async,
            )
            responses.append(result.generations[0][0].text)

        json_responses = [
            await json_loader.safe_load(item, self.llm, is_async=is_async)
            for item in responses
        ]
        json_responses = t.cast(t.List[t.Dict], json_responses)
        score = self._calculate_average_precision(json_responses)
        return score

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logging.info(f"Adapting Context Precision to {language}")
        self.context_precision_prompt = self.context_precision_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_precision_prompt.save(cache_dir)


@dataclass
class ContextUtilization(ContextPrecision):
    name: str = "context_utilization"
    evaluation_mode: EvaluationMode = EvaluationMode.qac

    def get_dataset_attributes(self, dataset: Dataset):
        return dataset["question"], dataset["contexts"], dataset["answer"]


context_precision = ContextPrecision()
context_utilization = ContextUtilization()
