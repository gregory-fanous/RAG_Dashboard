# TODO

## Next Stage

- Add optimization API endpoints to automatically search and tune best RAG parameters per dataset and use-case.
- Scope:
  - `POST /api/optimizations/run` to launch parameter search jobs.
  - `GET /api/optimizations` to list optimization runs and statuses.
  - `GET /api/optimizations/{run_id}` to inspect best configs, metric frontier, and tradeoffs.
  - `POST /api/optimizations/{run_id}/promote` to promote a winning config into workflow defaults/governance baselines.
