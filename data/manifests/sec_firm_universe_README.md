# SEC firm universe -- survivorship-bias audit trail

`sec_firm_universe.csv` lists **every** CIK returned by the SEC EDGAR
`browse-edgar` full-text-search atom feed for SIC code 3674 (Semiconductors
& Related Devices) with at least one 10-K filing on record, as of the run
that produced this file. It is **not** limited to firms currently in the
PHLX Semiconductor (SOX) index.

## Why the universe is broader than the SOX index

Restricting the sample to *current* SOX constituents would introduce
survivorship bias: firms that were delisted, acquired, went bankrupt, or
were simply removed from the index over 2006-2026 would be silently
excluded, biasing fundamentals-based forecasts toward surviving firms.
Per the study protocol, the full SIC 3674 filer list is retained instead,
including:

- firms never in the SOX index (e.g. small-cap or foreign-private-issuer
  semiconductor filers under SIC 3674),
- firms delisted, merged, or acquired during the sample window,
- shell or inactive filers with an SIC 3674 registration but sparse/no
  XBRL data (these fail the companyfacts fetch and are logged, not
  fabricated).

Downstream index-membership filtering, if needed for a specific analysis,
should be applied explicitly and documented as its own step -- not baked
into this ingestion layer.

## Columns

- `cik` -- 10-digit zero-padded SEC Central Index Key
- `ticker` -- current ticker from SEC's `company_tickers.json` bulk file,
  or backfilled from the submissions API; null if the firm has no active
  listed ticker on record
- `name` -- registrant name
- `sic` -- SIC code (constant, "3674", by construction of the query)
- `source_note` -- how this row's identity fields were resolved, and any
  caveats (e.g. "not in company_tickers.json", "submissions API 404")

## Coverage note

Of 505 distinct CIKs identified, 115 had a matching
entry in `company_tickers.json` (i.e. an active listed ticker at the time
of that bulk file's snapshot). The remainder are retained in the universe
with `ticker` possibly null -- this is expected for delisted/inactive/
foreign-private-issuer filers and is itself part of the audit trail, not
an error.
