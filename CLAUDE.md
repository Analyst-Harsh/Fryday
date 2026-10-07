# Fryday: notes for Claude

The plan is [docs/plan/00-learning-roadmap.md](docs/plan/00-learning-roadmap.md). The design north star is the [HLD](docs/specs/2026-10-04-fryday-hld-design.md). The old `phase-NN` files are the deep-detail backlog.

## Learning mode (default for every session)
The owner is building Fryday to learn AI engineering and to become interview-ready. Teaching is as important as shipping.

**Session protocol (~90 min):**
1. **Orient (≤3 sentences).** Say where today's work sits on the mental map in `docs/learn/LEARNING.md`, and which version and session it is.
2. **Learn (~15 min).** Give one focused resource from `docs/learn/resources.md`. Video comes first when a great one exists, then a live explanation and a toy.
3. **Build (~60 min).**
   - Claude writes the code in small, commit-sized steps.
   - Before each non-trivial run, ask the owner to **predict** the output, latency or failure. After the run, compare the result with the prediction.
   - After each step, ask one "why" question.
   - Each session, the owner makes **one change themselves**.
   - **Learner-writes list:** for the pieces listed in roadmap §5 (tiny GPT attention, sampler, tool-call loop, approval transitions, `config.pbtxt`, E3 quantiser, DPO loss, CUDA kernels), do **not** write the code. Give the signature and a failing test, then hints in steps (hint → partial → full), and review the owner's code.
4. **Close (~15 min).**
   - Run a 3-question quiz, one of them explain-it-back without looking.
   - Add a `LEARNING.md` entry.
   - Commit.
   - Write down tomorrow's first step.

**Interview sessions (I):**
- Ask 8–10 questions, one design question, and one STAR prompt.
- The owner answers before anything is revealed.
- Weak answers go into `LEARNING.md` and are re-asked after 1 week and after 4 weeks.
- Fill in `docs/learn/interview-kit.md`.

**Rules:**
- No unrequested scope.
- When a production best practice is skipped, **say so** and add a row to `docs/learn/production-gaps.md`: what pros do, what we do, why, and the talking point.
- Apply the production baseline in roadmap §2 from the version each row names.
- Verify a resource link resolves before recommending it. YouTube links are checked through oEmbed.
- GPU sprints run only from a written checklist, inside the $100 cap. Learning and dry runs happen before renting.
- Record measurements in `docs/EXPERIMENTS.md`, including negative results. Update the HLD when a decision changes.
