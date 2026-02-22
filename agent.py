import logging
import os
from typing import Literal, Optional

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tqdm import tqdm

from filings_helper import get_complete_filings, get_filing_content, CompleteFiling

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

class TopicCheck(BaseModel):
    # tri-state is often more reliable than forcing a hard boolean
    verdict: Literal["yes", "no", "unclear"] = Field(
        description="Whether the topic is discussed in the text."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="1–3 short quotes (verbatim) from the text supporting the verdict."
    )
    reasoning: str = Field(
        description="One sentence explaining the verdict, referencing the evidence."
    )


class FilingAnalysis(BaseModel):
    filing: CompleteFiling
    verdict: bool
    positive_topic_check: Optional[TopicCheck] = None


class FlattenedFilingAnalysis(CompleteFiling):
    verdict: Literal[0, 1]
    evidence: str = ""
    reasoning: str = ""

    @classmethod
    def from_analysis(cls, analysis: FilingAnalysis) -> "FlattenedFilingAnalysis":
        topic_check = analysis.positive_topic_check
        return cls.model_validate({
            **analysis.filing.model_dump(),
            "verdict": 1 if analysis.verdict else 0,
            "evidence": " | ".join(topic_check.evidence) if topic_check else "",
            "reasoning": topic_check.reasoning if topic_check else ""
        })


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


logging.getLogger("httpx").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

INSTRUCTION_TEMPLATE = (
    "You are checking whether a TOPIC is discussed in a TEXT CHUNK.\n"
    "Rules:\n"
    "- Only use the provided text; do not use outside knowledge.\n"
    "- Say 'yes' only if the topic is explicitly discussed (not just hinted).\n"
    "- If the topic is only vaguely alluded to, use 'unclear'.\n"
    "- Provide 1–3 short verbatim quotes from the text as evidence.\n"
    "- If verdict is 'no', evidence must be empty."
)

model_name = os.getenv("MODEL_NAME", "gpt-5.2")
embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-small")
model = OpenAIChatModel(model_name=model_name, provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))
agent = Agent(model=model, output_type=TopicCheck, instructions=INSTRUCTION_TEMPLATE, retries=5)

embed_model = OpenAIEmbedding(model=embeddings_model_name, api_key=os.getenv("OPENAI_API_KEY"))
splitter = SemanticSplitterNodeParser(embed_model=embed_model, buffer_size=1)

# Mag7
# TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

# Read tickers from oex.txt:
with open("oex.txt", "r") as f:
    TICKERS = f.read().splitlines()

FILING_TYPE: Literal["8-K", "10-K"] = "8-K"
LOGGER.info("Initialized with %s tickers...", len(TICKERS))


def topic_is_discussed(text_chunk: str, topic: str) -> tuple[TopicCheck, RunUsage]:
    prompt = (
        f"TOPIC:\n{topic}\n\n"
        f"TEXT CHUNK:\n{text_chunk}\n"
    )
    result = agent.run_sync(prompt)
    return result.output, result.usage()


filings: list[CompleteFiling] = []

# Ignore filings before 2022-01-01
start_date = "2022-01-01"

for ticker in tqdm(TICKERS, desc=f"Fetching {FILING_TYPE} filings"):
    all_filings = get_complete_filings(ticker, FILING_TYPE)
    filings.extend(filing for filing in all_filings if filing.filing_date >= start_date)

LOGGER.info("Fetched %s filings", len(filings))

overall_usage = Usage(input_tokens=0, output_tokens=0)
results: list[FilingAnalysis] = []

pbar = tqdm(filings, desc="Starting analysis...")


def get_description(i: int, total: int, usage: Usage) -> str:
    return f"{i + 1}/{total} ({usage.input_tokens} input tokens, {usage.output_tokens} output tokens)"


encoder = tiktoken.get_encoding("cl100k_base")
max_tokens = 4096

split = False
class Node(BaseModel):
    text: str

for filing in pbar:
    if split:
        tokens = encoder.encode(get_filing_content(filing))
        capped_token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        docs = [Document(text=encoder.decode(chunk)) for chunk in capped_token_chunks]
        nodes = splitter.get_nodes_from_documents(docs)
    else:
        nodes = [Node(text=get_filing_content(filing))]
    for i, node in enumerate(nodes):
        pbar.set_description(get_description(i, len(nodes), overall_usage))
        check, usage = topic_is_discussed(node.text, "AI")
        overall_usage.input_tokens += usage.input_tokens
        overall_usage.output_tokens += usage.output_tokens
        pbar.set_description(get_description(i, len(nodes), overall_usage))
        if check.verdict == "yes":
            results.append(FilingAnalysis(filing=filing, verdict=True, positive_topic_check=check))
            break
        results.append(FilingAnalysis(filing=filing, verdict=False))

LOGGER.info("Done! Overall usage report: %s", overall_usage)

results_csv_path = os.getenv("RESULTS_CSV_PATH", "filing_analysis_results.csv")
filing_fields = list(CompleteFiling.model_fields.keys())

flattened_results = [FlattenedFilingAnalysis.from_analysis(result).model_dump() for result in results]

(pd.DataFrame(flattened_results)
 .drop_duplicates(subset=filing_fields, keep="first")
 .reindex(columns=[*filing_fields, "verdict", "evidence", "reasoning"])
 .to_csv(results_csv_path, index=False))

LOGGER.info("Saved %s flattened results to %s", len(flattened_results), results_csv_path)
