import argparse
import json
import os
import sys
import time
from collections import Counter
from functools import cached_property
from itertools import chain, islice
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

import httpx
from markdownify import markdownify as md
from pydantic import BaseModel, ConfigDict

SEC_DELAY = float(os.getenv("SEC_DELAY", "0.1"))  # keep this small but non-zero


def _sec_get(url: str) -> httpx.Response:
    time.sleep(SEC_DELAY)
    headers = {
        "User-Agent": os.environ.get("SEC_USER_AGENT", "Example example@example.com"),
        "Accept-Encoding": "gzip, deflate",
    }
    return httpx.get(url, headers=headers, timeout=30).raise_for_status()


class Filing(BaseModel):
    model_config = ConfigDict(frozen=True)

    form: str
    accession: str
    filing_date: str
    primary_doc: str


class FilingsQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope: Literal["recent", "all"] = "recent"
    form: Optional[str] = None
    limit: Optional[int] = None

    @classmethod
    def from_params(
            cls,
            *,
            scope: str,
            form: Optional[str],
            limit: Optional[int],
    ) -> "FilingsQuery":
        if scope not in {"recent", "all"}:
            raise ValueError("scope must be 'recent' or 'all'")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")

        form_upper: Optional[str] = None
        if form is not None:
            value = form.strip()
            form_upper = value.upper() if value else None

        return cls(scope=scope, form=form_upper, limit=limit)


class SECFilingsClient:
    def __init__(self, fetch: Callable[[str], httpx.Response] = _sec_get) -> None:
        self._fetch = fetch

    def _get_json(self, url: str) -> Dict[str, Any]:
        return self._fetch(url).json()

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

    def iter_filing_pages(
            self,
            submissions: Dict[str, Any],
            *,
            scope: Literal["recent", "all"],
    ) -> Iterator[Dict[str, Any]]:
        yield submissions["filings"]["recent"]
        if scope == "recent":
            return

        files = (submissions.get("filings", {}) or {}).get("files", [])
        for meta in files:
            name = meta.get("name")
            if name:
                yield self._get_json(f"https://data.sec.gov/submissions/{name}")

    @staticmethod
    def iter_filing_rows(
            page: Dict[str, Any],
            *,
            form_upper: Optional[str],
    ) -> Iterator[Filing]:
        for filing_form, accession, filing_date, primary_doc in zip(
                page["form"],
                page["accessionNumber"],
                page["filingDate"],
                page["primaryDocument"],
        ):
            if form_upper is not None and filing_form.upper() != form_upper:
                continue

            yield Filing(
                form=filing_form,
                accession=accession,
                filing_date=filing_date,
                primary_doc=primary_doc,
            )

    def iter_filings(
            self, ticker_or_cik: str, *, query: FilingsQuery
    ) -> Iterator[Filing]:
        submissions = self.submissions(ticker_or_cik)
        rows = chain.from_iterable(
            self.iter_filing_rows(page, form_upper=query.form)
            for page in self.iter_filing_pages(submissions, scope=query.scope)
        )
        if query.limit is None:
            return rows
        return islice(rows, query.limit)

    def download_doc_text(
            self,
            ticker_or_cik: str,
            accession: str,
            primary_doc: str,
    ) -> str:
        cik = self.resolve_cik(ticker_or_cik)
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}/{primary_doc}"
        )
        return md(self._fetch(url).text)

    def get_filings(self,
                    ticker: str,
                    *,
                    scope: str = "recent",
                    form: Optional[str] = None,
                    limit: Optional[int] = None,
                    ) -> List[Dict[str, str]]:
        """Fetch filings for a ticker.

        Args:
            ticker: Ticker symbol or numeric CIK.
            scope: "recent" (default) for inline recent filings only, or "all" for
                recent + older pages referenced in `filings.files`.
            form: Optional form filter (case-insensitive).
            limit: Optional maximum number of results to return.

        Returns:
            List of filing dicts with keys: form, accession, filing_date, primary_doc.
        """
        query = FilingsQuery.from_params(scope=scope, form=form, limit=limit)
        filings = _CLIENT.iter_filings(ticker, query=query)
        return [filing.model_dump() for filing in filings]

    def filings_count_by_form(self, ticker: str, form: Optional[str] = None) -> Dict[str, int]:
        """Return counts grouped by form for a ticker (optionally filtered by form)."""
        filings = self.get_filings(ticker, scope="all", form=form)
        if form is not None and not filings:
            raise ValueError(f"no filings found for type {form}")

        counts: Counter[str] = Counter(f["form"] for f in filings)
        return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


_CLIENT = SECFilingsClient()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch SEC recent filings and latest filing content for a ticker or CIK"
    )
    parser.add_argument("ticker", help="Ticker symbol or numeric CIK")
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
                counts_dict = _CLIENT.filings_count_by_form(args.ticker, form=args.filing_type)
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
            latest["accession"],
            latest["primary_doc"],
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    latest_title = (
        "\nLatest filing content "
        f"(form={latest['form']}, filing_date={latest['filing_date']}, "
        f"accession={latest['accession']}, primary_doc={latest['primary_doc']}):"
    )

    print("Filings:")
    print(json.dumps(filings, indent=2))
    print(latest_title)
    print(latest_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
