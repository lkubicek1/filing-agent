import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()


def configure_logging() -> None:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


class TickerInfo(BaseModel):
    sector: str
    industry: str


configure_logging()

LOGGER = logging.getLogger(__name__)

model = OpenAIChatModel(
    "gpt-5.2", provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
)
agent = Agent(model=model, output_type=TickerInfo)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
LOGGER.info("Initialized with tickers: %s", TICKERS)

for ticker in TICKERS:
    LOGGER.info("Processing ticker: %s", ticker)
    result = agent.run_sync(f"What is the sector and industry of {ticker}?")
    LOGGER.info("Result for %s: %s | Usage: %s", ticker, result.output, result.usage())

LOGGER.info("Done!")
