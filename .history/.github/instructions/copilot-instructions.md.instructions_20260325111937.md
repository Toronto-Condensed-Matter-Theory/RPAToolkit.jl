---
description: Describe when these instructions should be loaded by the agent based on task context
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

Use minimal, deterministic solutions.
Do not add fallback behavior unless explicitly requested.
Do not add regex or broad file-search fallback logic to “recover” from missing inputs.
If inputs are missing or invalid, fail fast with a clear error message.
Prefer one explicit data flow over multiple implicit paths.
Do not mutate user input files unless explicitly requested.
When fixing errors, make the smallest direct fix first.
Avoid speculative retries and convoluted “just in case” branches.
Example: No fallbacks. Fail hard. Minimal patch only.