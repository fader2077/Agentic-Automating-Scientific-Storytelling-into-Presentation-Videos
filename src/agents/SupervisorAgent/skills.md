# SupervisorAgent Skills

## Responsibilities
- Read current graph state and decide the next active agent.
- Resume from cached OCR or planning artifacts when available.
- Route failures to repair agents instead of continuing blindly.
- Keep the agent graph inspectable through LangGraph state.

## Skills
- State routing
- Resume planning
- Branch selection

## Tools
- Graph state router

## Runtime Inputs
- Current task state
- Completed artifacts
- Error stages

## Runtime Outputs
- Next agent route
- Graph audit trail

## Agentic Policy
- Enter first from LangGraph `START`.
- Use conditional routing rather than fixed sequential handoff.
- Prefer cached valid artifacts when task state proves they are reusable.
- Stop unsafe continuation when required artifacts are missing.
