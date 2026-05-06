# AGENT_GUIDE.md — How to use the agent files (for humans)

This is for **you**, not for the AI tool. Read it once. It is short on purpose.

---

## Where these files live

- `AGENTS.md` is at the **repo root**. It is the entry point that AI tools find
  automatically (most modern agentic tools look for `AGENTS.md` at the repo
  root by convention).
- The other documents live in this `agent/` folder: `PHASES.md`, `STATUS.md`,
  `QAT_REPORT.md`, and this guide.
- `AGENTS.md` at the root has links into `agent/`. So a tool that reads
  `AGENTS.md` first will be told where to find everything else.

---

## The four files

1. **`AGENTS.md`** (repo root) — Project context for any AI tool. Tells the
   tool what this project is, how the code is organized, what the rules are,
   and how to work efficiently. You should rarely need to edit it. Update it
   only when something fundamental changes (build system, language, repo
   conventions).

2. **`agent/PHASES.md`** — The four-phase plan with entry and exit criteria.
   Edit this when the plan changes. Do not edit it casually.

3. **`agent/STATUS.md`** — Where the project is right now. **This is the file
   you update most often.** One-line entries when something changes. Top of
   the file is the current state; bottom is a recent-changes log.

4. **`agent/QAT_REPORT.md`** — The technical analysis of the current
   quantization implementation. Update only when the implementation changes
   meaningfully.

---

## How to start a new conversation with any AI tool

Just say one sentence:

> "We're working on the QAT GPT-2.c project. See `AGENTS.md` and `STATUS.md`.
> Today I want to: \<one specific thing\>."

That's it. The tool reads those files and is oriented. If the tool ignores
those files, point it at them explicitly.

---

## How to keep the files useful

**After every meaningful change**, do this:

- Add a one-line entry under "Recent changes log" in `STATUS.md` with today's
  date. Example: `2026-05-02 — Fixed master init bug; FP32 master now copied
  from original weights before PTQ compaction.`
- Move items between the "What is done", "What is in progress", and "What is
  next" sections in `STATUS.md` as appropriate.
- If you finish a phase, update `STATUS.md` to point at the new current phase
  and mark the old one done.

If you don't keep `STATUS.md` current, the next AI session won't know where
you are and will give you stale advice.

---

## When the AI is wasting your time

If the AI tool is re-reading the same large file every conversation, or
re-discovering the project structure, or asking you the same setup questions:
that is the symptom. The cause is usually that `STATUS.md` is out of date or
the tool didn't read `AGENTS.md`.

Try this in order:

1. Tell the tool: "Read `AGENTS.md` and `STATUS.md` first."
2. If that doesn't work, paste the relevant section of `STATUS.md` into the
   conversation directly.
3. If the tool still goes off-track, ask it to point to the file and line
   that justifies its claim. That usually surfaces where it's hallucinating.

---

## When you change phase

You finish Phase 1 (the exit criteria are all met):

1. In `STATUS.md`, change "Current phase" to Phase 2.
2. Move the Phase 1 outstanding work list to a new "Phase 1 — done" section
   (or just delete it; git remembers).
3. Copy the Phase 2 entry/exit criteria from `PHASES.md` into a new "Phase 2
   outstanding work" section in `STATUS.md`.
4. Add a recent-changes log entry: `2026-XX-XX — Phase 1 closed. Entering
   Phase 2 (COAT optimizer state compression).`

---

## When the plan changes

If you change the plan (add a phase, drop a phase, change scope):

1. Update `PHASES.md` first.
2. Then update `STATUS.md` to reflect the new current phase.
3. Add a recent-changes log entry explaining why.

---

## What to do when an AI suggests big changes

The AI might suggest "let me refactor X" or "let me change the model file
format". Default answer: **no, not in this turn.** Ask it to:

1. Write down the proposed change in `STATUS.md` under "Open questions".
2. Wait for you to decide.

Big changes (format, dependencies, architecture) belong in a separate decision
step, not folded into a feature change.

---

## TL;DR

- `AGENTS.md` = rules. Rarely edited.
- `PHASES.md` = plan. Occasionally edited.
- `STATUS.md` = current state. Edited often.
- `QAT_REPORT.md` = analysis. Edited when the code changes meaningfully.

Start each AI session with: "See `AGENTS.md` and `STATUS.md`. Today I want
to: ..."

End each meaningful change with: one-line entry in `STATUS.md`'s recent
changes log.

That's the whole system.
