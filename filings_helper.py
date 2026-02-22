import argparse
import json
import os
import sys
import time
from collections import Counter
from functools import cached_property
from itertools import islice
from typing import Annotated, Any, Dict, List, Literal, Optional

import httpx
from markdownify import markdownify as md
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    StringConstraints,
    validate_call,
)

SEC_DELAY = float(os.getenv("SEC_DELAY", "0.1"))  # keep this small but non-zero


class Filing(BaseModel):
    model_config = ConfigDict(frozen=True)

    form: str
    accession: str
    filing_date: str
    primary_doc: str


class SECFilingsClient:

    @staticmethod
    def _sec_get(url: str) -> httpx.Response:
        time.sleep(SEC_DELAY)
        headers = {
            "User-Agent": os.environ.get("SEC_USER_AGENT", "Example example@example.com"),
            "Accept-Encoding": "gzip, deflate",
        }
        return httpx.get(url, headers=headers, timeout=30).raise_for_status()

    def _get_json(self, url: str) -> Dict[str, Any]:
        return self._sec_get(url).json()

    @cached_property
    def ticker_to_cik(self) -> Dict[str, int]:
        data = self._get_json("https://www.sec.gov/files/company_tickers.json")
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}

    def resolve_cik(self, ticker: str) -> int:
        try:
            return self.ticker_to_cik[ticker.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown ticker: {ticker}") from exc

    def submissions(self, ticker: str) -> Dict[str, Any]:
        cik = str(self.resolve_cik(ticker)).zfill(10)
        return self._get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

    def download_doc_text(
            self,
            ticker: str,
            accession: str,
            primary_doc: str,
    ) -> str:
        cik = self.resolve_cik(ticker)
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}/{primary_doc}"
        )
        return md(self._sec_get(url).text)

    @validate_call
    def get_filings(
            self,
            ticker: str,
            *,
            scope: Literal["recent", "all"] = "recent",
            form: Annotated[
                Optional[str], StringConstraints(strip_whitespace=True, to_upper=True)
            ] = None,
            limit: Optional[PositiveInt] = None,
    ) -> List[Filing]:
        """Fetch filings for a ticker.

        Args:
            ticker: Ticker symbol.
            scope: "recent" (default) for inline recent filings only, or "all" for
                recent + older pages referenced in `filings.files`.
            form: Optional form filter (case-insensitive).
            limit: Optional maximum number of results to return.

        Returns:
            List of strongly-typed `Filing` models.
        """
        submissions = self.submissions(ticker)

        def pages():
            yield submissions["filings"]["recent"]
            if scope == "recent": return
            base = "https://data.sec.gov/submissions"
            yield from (
                self._get_json(f"{base}/{name}")
                for m in submissions.get("filings", {}).get("files", ())
                if (name := m.get("name"))
            )

        target_form = form.upper() if form else None

        rows = (
            Filing(form=f, accession=a, filing_date=d, primary_doc=p)
            for page in pages()
            for f, a, d, p in zip(
            page.get("form", ()),
            page.get("accessionNumber", ()),
            page.get("filingDate", ()),
            page.get("primaryDocument", ()),
        )
            if not target_form or f.upper() == target_form
        )

        return list(islice(rows, limit)) if limit is not None else list(rows)

    def filings_count_by_form(
            self, ticker: str, form: Optional[str] = None
    ) -> Dict[str, int]:
        """Return counts grouped by form for a ticker (optionally filtered by form)."""
        filings = self.get_filings(ticker, scope="all", form=form)
        if form is not None and not filings:
            raise ValueError(f"no filings found for type {form}")

        counts: Counter[str] = Counter(f.form for f in filings)
        return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


_CLIENT = SECFilingsClient()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch SEC recent filings and latest filing content for a ticker"
    )
    parser.add_argument("ticker", help="Ticker symbol")
    parser.add_argument(
        "--type",
        dest="filing_type",
        help="Filter by form type (example: 8-K)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent filings to list (default: 10)",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Count all filings grouped by form type",
    )
    args = parser.parse_args()

    if args.limit < 1:
        print("Error: --limit must be >= 1", file=sys.stderr)
        return 2

    try:
        if args.counts:
            try:
                counts_dict = _CLIENT.filings_count_by_form(
                    args.ticker, form=args.filing_type
                )
            except ValueError:
                if args.filing_type:
                    print(
                        f"Error: no filings found for type {args.filing_type}",
                        file=sys.stderr,
                    )
                else:
                    print("Error: no filings found", file=sys.stderr)
                return 1

            print("Filing counts by type:")
            print(json.dumps(counts_dict, indent=2))
            print(f"Total filings: {sum(counts_dict.values())}")
            return 0

        filings = _CLIENT.get_filings(
            args.ticker,
            scope="recent",
            form=args.filing_type,
            limit=args.limit,
        )
        if not filings:
            msg = "Error: no recent filings found"
            if args.filing_type:
                msg = f"Error: no recent filings found for type {args.filing_type}"
            print(msg, file=sys.stderr)
            return 1

        latest = filings[0]
        latest_content = _CLIENT.download_doc_text(
            args.ticker,
            latest.accession,
            latest.primary_doc,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    latest_title = (
        "\nLatest filing content "
        f"(form={latest.form}, filing_date={latest.filing_date}, "
        f"accession={latest.accession}, primary_doc={latest.primary_doc}):"
    )

    print("Filings:")
    print(json.dumps([filing.model_dump() for filing in filings], indent=2))
    print(latest_title)
    print(latest_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
