# ArtifactJoinAgent Skills

## Responsibilities
- Synchronize parallel slide and speech branches.
- Gate cursor grounding until both `slide_imgs` and `audio` artifacts exist.
- Prevent render work from starting with partial artifacts.

## Skills
- Parallel branch synchronization
- Artifact readiness checks
- Handoff gating

## Tools
- Graph state join

## Runtime Inputs
- Slide branch state
- Speech branch state
- Artifact list

## Runtime Outputs
- Grounding route
- Wait state when a branch is incomplete

## Agentic Policy
- Enter from both `VisualAuditorAgent` and `SpeechAgent`.
- Use conditional join routing, not fixed order.
- Continue to `GroundingAgent` only when required artifacts are present.
