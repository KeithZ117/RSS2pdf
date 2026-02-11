# RSS2pdf

A minimal RSS/Atom to PDF script with zero third-party dependencies.

## Features

- Reads feed sources from `feeds.opml`
- Fetches unseen entries from RSS/Atom feeds
- Creates a daily output folder: `output/YYYY-MM-DD/`
- Generates:
  - `rss_digest.pdf`
  - `run.json` (run summary + fetch errors)
- Prevents duplicate output across runs using `.rss2pdf/state.json`

## Run Locally

```bash
python rss_to_pdf.py
```

Common example:

```bash
python rss_to_pdf.py --opml feeds.opml --max-per-feed 5 --timeout 10 --workers 8
```

## CLI Options

- `--opml`: OPML file path (default: `feeds.opml`)
- `--output-dir`: Output root directory (default: `output`)
- `--state-file`: Dedup state file (default: `.rss2pdf/state.json`)
- `--max-per-feed`: Max unseen entries per feed
- `--timeout`: Per-feed request timeout (seconds)
- `--workers`: Number of concurrent feed fetch workers
- `--summary-chars`: Max summary text length per entry

## Deduplication

Each exported entry is fingerprinted and stored in `.rss2pdf/state.json`.
Future runs skip entries already recorded there.

## GitHub Actions (Scheduled Runs)

Workflow file: `.github/workflows/rss-fetch.yml`

Current behavior:

- Manual trigger: `workflow_dispatch`
- Scheduled trigger: `cron: 0 1 * * *` (01:00 UTC daily)
- Auto-commits `output/` and `.rss2pdf/state.json` back to the repository

To change schedule, edit the `cron` expression in `.github/workflows/rss-fetch.yml`.
