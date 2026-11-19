# AGENTS.md — Instructions for Codex and other AI agents

## Project context

This is a hands-on learning repository for Gen AI concepts (RAG, embeddings, chunking, document extraction, etc.). The code is written to teach — comments are as important as the code itself.

## Comment preservation (mandatory)

Developer comments in this codebase explain *why* and *how* things work. They exist for human learners, not just for the machine. When editing any file:

1. **Never remove existing comments.** They are intentional teaching material.
2. **If you change code behavior, update the nearby comments** to reflect the new behavior. Do not leave stale comments.
3. **If you add new code that has no comments, add concise intuitive comments** explaining the intent — do not introduce uncommented code next to commented code.
4. **Do not replace commented code with uncommented code.** The replacement must carry forward the same level of explanation.

This applies to all comment styles: inline `#`, docstrings, block comments, and Markdown explanations embedded in notebooks.

## File metadata convention

`demo_sample_data/unstructured_data/raw/_metadata.json` must be updated whenever a new document is added to `raw/`. See `demo_sample_data/unstructured_data/USAGE.md` for the full field reference.
