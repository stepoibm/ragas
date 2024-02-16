import os
import pandas as pd
from pathlib import Path
import dotenv
from ragas.metrics import context_precision, faithfulness, context_recall, answer_relevancy
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ragas import evaluate
from datasets import Dataset

from langchain_watson_bam import LangchainWatsonBam

metrics = [
    context_precision,
    #faithfulness,
    #context_recall,
    #answer_relevancy,
]

dotenv.load_dotenv()

input_folder = f'./'
output_folder = f'./'

milvus_result_files = list(Path(input_folder).glob("*05*milvus*.csv"))
milvus_result_file = max(milvus_result_files, key=os.path.getctime)
df_result = pd.read_csv(milvus_result_file)

wai_api_key = os.getenv('WAI_API_KEY')
wai_model_id = os.getenv('WAI_MODEL_ID')
wai_project_id = os.getenv('WAI_PROJECT_ID')
wai_url = os.getenv("WAI_URL")


model_name = "intfloat/multilingual-e5-large"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
sentence_transformers_embedding = SentenceTransformerEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

parameters = {
    "min_new_tokens": 0,
    "max_new_tokens": 2000,
    "decoding_method": "greedy",
    "repetition_penalty": 1,
}

watsonx_llm = LangchainWatsonBam(
    model_id=wai_model_id,
    url=wai_url,
    apikey=wai_api_key,
    params=parameters,
    max_workers=6,
)


df_dataset = df_result.rename(
    columns={
        "golden_question": "question",
        "golden_answer": "ground_truth",
        "retrieved_generated": "answer",
        "golden_context": "contexts",
    }
)

df_dataset["contexts"] = df_dataset["contexts"].apply(
    lambda x: [x] if isinstance(x, str) else []
)

d = {
    "question": df_dataset["question"].tolist(),
    "answer": df_dataset["answer"].tolist(),
    "ground_truth": df_dataset["ground_truth"].tolist(),
    "contexts": df_dataset["contexts"].tolist(),
}

dataset = Dataset.from_dict(d)


if __name__ == '__main__':
    evaluation_result = evaluate(
        dataset,
        is_async=True,
        metrics=metrics,
        llm=watsonx_llm,
        embeddings=sentence_transformers_embedding,
        raise_exceptions=False,
    )

    df = evaluation_result.to_pandas()
    df.to_csv("07_ragas_evaluation.csv")