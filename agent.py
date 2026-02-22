import logging
import os
import random
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

def format_tokens(n: int) -> str:
    if n < 100_000:
        return f"{n:,}"
    elif n < 1_000_000:
        return f"{round(n / 1_000)}K"
    else:
        # Show one decimal only if needed
        value = n / 1_000_000
        if value.is_integer():
            return f"{int(value)}M"
        return f"{value:.1f}M"


class TopicCheck(BaseModel):
    # tri-state is often more reliable than forcing a hard boolean
    verdict: Literal["yes", "no", "unclear"] = Field(
        description="Whether the topic is discussed in the text."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="1–3 short quotes (verbatim) from the text supporting the verdict.",
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
        return cls.model_validate(
            {
                **analysis.filing.model_dump(),
                "verdict": 1 if analysis.verdict else 0,
                "evidence": " | ".join(topic_check.evidence) if topic_check else "",
                "reasoning": topic_check.reasoning if topic_check else "",
            }
        )


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
model = OpenAIChatModel(
    model_name=model_name, provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
)
agent = Agent(
    model=model, output_type=TopicCheck, instructions=INSTRUCTION_TEMPLATE, retries=5
)

embed_model = OpenAIEmbedding(
    model=embeddings_model_name, api_key=os.getenv("OPENAI_API_KEY")
)
splitter = SemanticSplitterNodeParser(embed_model=embed_model, buffer_size=1)

# Mag7
# TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

# Read tickers from oex.txt:
with open("oex.txt", "r") as f:
    TICKERS = f.read().splitlines()
random.shuffle(TICKERS)

FILING_TYPE: Literal["8-K", "10-K"] = "8-K"
LOGGER.info("Initialized with %s tickers...", len(TICKERS))


def topic_is_discussed(text_chunk: str, topic: str) -> tuple[TopicCheck, RunUsage]:
    prompt = f"TOPIC:\n{topic}\n\nTEXT CHUNK:\n{text_chunk}\n"
    result = agent.run_sync(prompt)
    return result.output, result.usage()


encoder = tiktoken.get_encoding("cl100k_base")
filings: list[CompleteFiling] = []

# Ignore filings before 2022-01-01
start_date = "2022-01-01"

pbar = tqdm(TICKERS, desc=f"Starting to fetch {FILING_TYPE} filings...")
pbar.set_postfix(collected_filings=0, failed_tickers=0)
failed_tickers: list[str] = []
for ticker in pbar:
    pbar.set_description(f"Fetching {FILING_TYPE} filings for <{ticker}> since {start_date} ")
    try:
        filings.extend(get_complete_filings(ticker=ticker, doc_type=FILING_TYPE, start_date=start_date))
    except Exception as exc:
        failed_tickers.append(ticker)
        continue
    pbar.set_postfix(collected_filings=len(filings), failed_tickers=len(failed_tickers))

LOGGER.info("Fetched %s filings", len(filings))

results_jsonl_path = os.getenv("RESULTS_JSONL_PATH", "filing_analysis_results.jsonl")


def load_persisted_results(path: str) -> list[FilingAnalysis]:
    if not os.path.exists(path):
        return []

    persisted_results: list[FilingAnalysis] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                persisted_results.append(FilingAnalysis.model_validate_json(payload))
            except Exception as exc:
                LOGGER.warning(
                    "Skipping invalid line %s in %s: %s", line_number, path, exc
                )

    return persisted_results


persisted_results = load_persisted_results(results_jsonl_path)
processed_filing_keys = {result.filing.hash for result in persisted_results}

pending_filings: list[CompleteFiling] = []
seen_filing_keys = set(processed_filing_keys)
for filing in filings:
    if filing.hash in seen_filing_keys:
        continue
    seen_filing_keys.add(filing.hash)
    pending_filings.append(filing)

LOGGER.info(
    "Loaded %s persisted results from %s; skipping %s filings",
    len(persisted_results),
    results_jsonl_path,
    len(filings) - len(pending_filings),
)

overall_usage = Usage(input_tokens=0, output_tokens=0)
results: list[FilingAnalysis] = list(persisted_results)


def get_description(i: int, total: int, usage: Usage) -> str:
    return f"{i + 1}/{total} ({usage.input_tokens} input tokens, {usage.output_tokens} output tokens)"


def persist_result(result: FilingAnalysis) -> None:
    with open(results_jsonl_path, "a", encoding="utf-8") as f:
        f.write(result.model_dump_json())
        f.write("\n")


max_tokens = 4096

split = False


class Node(BaseModel):
    text: str


pbar = tqdm(pending_filings, desc="Starting token estimation ...")
estimated_input_tokens = 0
pbar.set_postfix(estimated_input_tokens=format_tokens(estimated_input_tokens))
filings_with_content: list[tuple[CompleteFiling, str]] = []
for filing in pbar:
    pbar.set_description(f"Fetching content for {filing.ticker}...")
    filing_content = get_filing_content(filing)
    filings_with_content.append((filing, filing_content))
    tokens = encoder.encode(filing_content)
    estimated_input_tokens += len(tokens)
    pbar.set_postfix(estimated_input_tokens=format_tokens(estimated_input_tokens))

# The agent will scan each filing for mentions of the following topic
TOPIC_TO_CHECK = "AI"

pbar = tqdm(filings_with_content, desc="Starting agent analysis...")
pbar.set_postfix(total_input_tokens=overall_usage.input_tokens, total_output_tokens=overall_usage.output_tokens)
for filing, content in pbar:
    pbar.set_description(f"Agent is analyzing [{filing.ticker}:{filing.doc_type} {filing.filing_date}]")
    if split:
        tokens = encoder.encode(content)
        capped_token_chunks = [
            tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)
        ]
        docs = [Document(text=encoder.decode(chunk)) for chunk in capped_token_chunks]
        nodes = splitter.get_nodes_from_documents(docs)
    else:
        nodes = [Node(text=content)]
    for i, node in enumerate(nodes):
        pbar.set_postfix(node=f"{i+1}/{len(nodes)}", total_input_tokens=overall_usage.input_tokens, total_output_tokens=overall_usage.output_tokens)
        node_text = node.text if isinstance(node, Node) else node.get_content()
        check, usage = topic_is_discussed(node_text, TOPIC_TO_CHECK)
        overall_usage.input_tokens += usage.input_tokens
        overall_usage.output_tokens += usage.output_tokens
        pbar.set_description(get_description(i, len(nodes), overall_usage))
        if check.verdict == "yes":
            analysis = FilingAnalysis(
                filing=filing, verdict=True, positive_topic_check=check
            )
            results.append(analysis)
            persist_result(analysis)
            break
        analysis = FilingAnalysis(filing=filing, verdict=False)
        results.append(analysis)
        persist_result(analysis)

LOGGER.info("Done! Overall usage report: %s", overall_usage)
LOGGER.info("Saved incremental results to %s", results_jsonl_path)

results_csv_path = os.getenv("RESULTS_CSV_PATH", "filing_analysis_results.csv")
filing_fields = list(CompleteFiling.model_fields.keys())

flattened_results = [
    FlattenedFilingAnalysis.from_analysis(result).model_dump() for result in results
]

(
    pd.DataFrame(flattened_results)
    .drop_duplicates(subset=filing_fields, keep="first")
    .reindex(columns=[*filing_fields, "verdict", "evidence", "reasoning"])
    .to_csv(results_csv_path, index=False)
)

LOGGER.info(
    "Saved %s flattened results to %s", len(flattened_results), results_csv_path
)
