# Release validation: 0.1.0

Validated on macOS with Python 3.12 on 21 September 2026.

- The public tree completed 121 unit tests, with two skips: the unavailable private frozen-run fixture and a newer Anthropic SDK protocol check. Provider request tests use offline responses.
- The wheel built successfully and installed with its base dependencies in a new virtual environment.
- The supplied-topic offline workflow verifies export, save/load, and individual-versus-batch assignment.
- Public coded predictions reproduce all 240 method configurations and 48 summary groups. Numerical summaries agree with the manuscript evidence within floating-point tolerance.
- Public package source matches the current model implementation. Credentials, original corpus text, research caches, logs and prior Git history are excluded.
- The paper includes the compact-prompt main experiment and complete neutral-prompt sensitivity results. Raw generation replay requires the original research artifacts described in the reproduction guide.

The two released main cloud experiments provide account-backed OpenAI evidence. Other provider adapters retain the validation scope described in README.md. No additional paid API experiment was run for this packaging step.
