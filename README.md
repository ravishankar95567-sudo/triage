 Multi-Domain Support Triage Agent

## Overview
This project is a terminal-based support triage agent that processes support tickets across three ecosystems:

- HackerRank
- Claude
- Visa

The agent classifies incoming support issues, determines risk, retrieves the most relevant support guidance from a support corpus, and decides whether to reply directly or escalate.

## Features
- Terminal-based execution
- CSV input processing
- Request type classification
- Product area classification
- Risk assessment
- Escalation logic
- Support retrieval using BM25
- Grounded response generation
- CSV output generation

## Tech Stack
- Python
- pandas
- rank-bm25

## Files
- triage.py — main support triage pipeline
- README.md — setup and approach documentation

## Setup
Install dependencies:

pip install pandas rank-bm25

## Run
python triage.py --input support_tickets.csv --output predictions.csv

## Output
The script generates:

predictions.csv

with the following columns:

- status
- product_area
- response
- justification
- request_type

## Approach
1. Load support issues from CSV
2. Normalize ticket input
3. Infer domain (HackerRank / Claude / Visa)
4. Classify request type
5. Assess issue risk
6. Retrieve relevant support guidance
7. Decide reply vs escalate
8. Generate grounded response
9. Export predictions.csv

## Safety
- Escalates high-risk issues
- Avoids unsupported claims
- Uses only support-corpus grounded responses