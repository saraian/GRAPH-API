
## Cross-agent handoff

Shared state for every agent (claude / opencode / antigravity) lives in
`.handoff/`. **Read `.handoff/HANDOFF.md` before doing anything**, and keep it
current so a handoff is possible at any moment. Plan and open work live in
`.handoff/plan/` (see `.handoff/plan/RULES.md`); closed work moves to
`.handoff/plan/bookkeeping/`.
To switch agent: `agent_handoff <this-agent> <other-agent>`.
