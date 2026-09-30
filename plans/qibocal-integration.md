# Completing the Qibocal notebook checkpoint

The converted platform and eight Qibocal acquisition/fit/report/update workflows
are implemented. The executed notebook export and all 123 targeted regressions
pass. Finish this work without expanding the physical model or adding automated
platform migration.

1. Review the checkpoint for accidental artifacts and confirm the notebook
   resets acquisition/update controls when its platform/data change. Retain the
   explicit limitations around binary qutrit readout, short coherence windows,
   coupled-system frequency shifts, and scalar T1/T2.
2. Delete this file immediately when the remaining work is complete, then
   commit and push the completion checkpoint.

A repository-wide Ruff invocation also reports a pre-existing import-order
issue in `tests/test_results.py`; the files changed for this integration pass
their focused checks. Do not confuse that baseline issue with a new regression.
