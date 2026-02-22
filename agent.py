import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


class TickerInfo(BaseModel):
    sector: str
    industry: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


LOGGER = logging.getLogger(__name__)

model_name = os.getenv("MODEL_NAME", "gpt-5.2")
model = OpenAIChatModel(model_name=model_name, provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))
agent = Agent(model=model, output_type=TickerInfo)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
LOGGER.info("Initialized with tickers: %s", TICKERS)

overall_usage = Usage(input_tokens=0, output_tokens=0)

for ticker in TICKERS:
    LOGGER.info("Processing ticker: %s", ticker)
    result = agent.run_sync(f"What is the sector and industry of {ticker}?")
    LOGGER.info("Result for %s: %s | Usage: %s", ticker, result.output, result.usage())
    overall_usage.input_tokens += result.usage().input_tokens
    overall_usage.output_tokens += result.usage().output_tokens

LOGGER.info("Done! Overall usage report: %s", overall_usage)
