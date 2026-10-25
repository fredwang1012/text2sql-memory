# text2sql-memory

A text-to-SQL layer that retrieves business logic, not just schema: metric definitions, join paths, naming
conventions and verified example queries. It bootstraps that knowledge from query logs and docs, and learns from use
by distilling user corrections into a reviewable, versioned memory. Fine-tuning is reserved for the retriever, never
for the business logic itself.

## Plan

| Stage | Done when |
| --- | --- |
| MVP | A business-knowledge store plus retrieval beats a schema-only baseline on execution accuracy on BIRD dev, with the BIRD evidence hidden. |
| Learning | Accuracy climbs over a simulated stream of user corrections (a memory curve). |
| Schema editor | An LLM drafts table and column descriptions for one BIRD database, editable in a UI; a setup chatbot asks about ambiguous columns and writes the answers back; chat-refined descriptions beat auto-drafted ones. |
| Serving | MCP access, and cost-aware execution (EXPLAIN checks, rewrites, learned efficient patterns) improves VES and estimated cost without losing accuracy. |
| Stretch | A retriever trained on usage logs beats off-the-shelf embeddings; writeup with ablations (no memory, flat RAG, typed memory, typed memory plus graph). |

## Status

Not started.

## License

Copyright (c) 2026 Frederick Wang. All rights reserved. See [LICENSE](LICENSE).
