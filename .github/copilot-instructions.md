# Project Rules — Cara Skin Analysis

## What This Project Is

A final degree project in Deep Learning. A web application that takes a facial photo/selfie and delivers a truthful report about the user's skin condition with ingredient recommendations. Multiple EfficientNetB0 models work in synergy, each detecting a different skin aspect, combined via a rule-based BLP engine.

## Non-Negotiable Rules

1. **Never guess.** Every answer must be based on facts from the code, data, or verified documentation. If unsure — ask questions.
2. **Never fake results.** If a model doesn't work well, say so. Don't manipulate thresholds or outputs to make things "look" correct.
3. **Truthfulness is the product.** The app must give reports that are actually true. A wrong prediction is worse than no prediction.
4. **Every change needs a test.** No exceptions. If you change code, prove it works.
5. **Keep a working version.** Before breaking changes, ensure we can roll back.

## Communication Style

- English only
- Plain language — no unnecessary jargon, but not dumbed down either
- Medium responses: include the "why" behind decisions
- When explaining DL concepts, use practical analogies
- If referencing papers, explain what they mean for our project in plain words

## Code Change Protocol

1. **Explain** what you plan to change and why
2. **Implement** the change
3. **Test** — write or run a test proving it works
4. **Verify** — run the test suite, confirm nothing else broke

## When Something Isn't Working

1. Ask what the user sees (symptoms)
2. Trace the pipeline from input to output
3. Find the exact point of failure (file + line + data)
4. Explain the root cause in plain language
5. Fix it with evidence, not hope
6. Add a test that would have caught this

## Priority Order

1. **Correctness** — does it produce the right answer?
2. **Clean code** — is it readable and maintainable?
3. **Speed** — is it fast enough? (optimize only when needed)

## Architecture Decisions

- **Backbone**: EfficientNetB0 (default). Change only with compelling evidence.
- **Fusion**: Rule-based BLP engine (not learned). Keep it deterministic and explainable.
- **Ingredients**: JSON rules mapping severity → recommendations. Keep rule-based.
- **Storage**: JSON files for now. MongoDB prepared but not connected.
- **Training**: CPU only. Keep models and batch sizes small.

## Error Handling

- **Model layer**: Fail loudly with clear error messages. Log tensor shapes and confidence values.
- **API layer**: Return structured error responses with helpful messages. Never expose raw tracebacks to users.
- **UI layer**: Show user-friendly messages. Guide user to retry (better lighting, clearer photo).
- **Pipeline**: If face not detected → clear message. If confidence too low → honest "unsure" response.

## Testing Standards

- Tests must verify **behavior**, not just "no crash"
- Each test tests ONE thing
- Test names describe the contract: `test_<component>_<scenario>_<expected_outcome>`
- Use real ground-truth images for model validation
- Mock external dependencies in unit tests
- Integration tests verify component connections

## Data & Training Rules

- Never train and evaluate on the same data
- Always maintain train/val/test split (70/15/15)
- Address class imbalance (clear is underrepresented) with weighted loss
- Augmentation on training data ONLY, never on validation/test
- Save best model by VALIDATION loss, not training loss
- When a model performs poorly, investigate data first

## Proactive Behavior

- Suggest improvements when you notice problems
- Flag potential issues even if not asked
- If you see code that could silently fail, raise it
- If a model change could affect other models, mention it

## Dependencies

- Add new packages freely if they solve real problems
- Prefer well-maintained, popular libraries
- Document why a dependency was added

## Git Practices

- Commit after each feature or fix
- Commit messages: `type: short description` (feat:, fix:, test:, refactor:, docs:)
- Never force-push or rewrite history without asking

## What The Agent Must NEVER Do

- Lower confidence thresholds to make bad predictions "pass"
- Generate synthetic results or fake evaluation metrics
- Claim something works without testing it
- Make changes without being able to explain why
- Add models without verified dataset quality
- Ignore class imbalance or data quality issues
- Skip the validation/test split for "convenience"
