"""Harnesses that mediate cross-cutting runtime behavior.

Harness modules keep policy-heavy logic out of task runners and model clients.
Each harness owns one runtime concern, such as prompt preparation or future
context-window management.
"""
