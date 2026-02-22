import logging
import os
from typing import Literal, Optional

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
agent = Agent(model=model, output_type=TopicCheck, instructions=INSTRUCTION_TEMPLATE)

embed_model = OpenAIEmbedding(model=embeddings_model_name, api_key=os.getenv("OPENAI_API_KEY"))
splitter = SemanticSplitterNodeParser(embed_model=embed_model, buffer_size=1)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
FILING_TYPE: Literal["8-K", "10-K"] = "10-K"
LOGGER.info("Initialized with tickers: %s", TICKERS)


def topic_is_discussed(text_chunk: str, topic: str) -> tuple[TopicCheck, RunUsage]:
    prompt = (
        f"TOPIC:\n{topic}\n\n"
        f"TEXT CHUNK:\n{text_chunk}\n"
    )
    result = agent.run_sync(prompt)
    return result.output, result.usage()


filings: list[CompleteFiling] = []

for ticker in tqdm(TICKERS, desc=f"Fetching {FILING_TYPE} filings"):
    filings.extend(get_complete_filings(ticker, "10-K"))

LOGGER.info("Fetched %s filings", len(filings))

overall_usage = Usage(input_tokens=0, output_tokens=0)
results: list[FilingAnalysis] = []

pbar = tqdm(filings, desc="Starting analysis...")


def get_description(i: int, total: int, usage: Usage) -> str:
    return f"{i + 1}/{total} ({usage.input_tokens} input tokens, {usage.output_tokens} output tokens)"


for filing in pbar:
    doc = Document(text=get_filing_content(filing))
    nodes = splitter.get_nodes_from_documents([doc])
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
