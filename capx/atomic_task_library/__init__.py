"""Long-term **atomic task** library (self-evolve fixed memory).

One module per atomic task = its function + config (hyper-parameters). These are
promoted, validated capabilities that are *not* bound to a specific object/task
(``docs-se/concepts.md`` §8). They are exposed to the agent by
``capx.self_evolve.long_term_library`` (docs in the prompt + injected into the
code-execution namespace), not by importing this package directly.
"""
