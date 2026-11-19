# Contributing

Run `python -m pytest -q`, `python -m resnet_smote smoke`, and `python -m build`.
Describe the changed behavior and validation in a focused pull request.

Preserve model semantics. Changes to splitting, resampling, model selection,
or label handling should include regression coverage using synthetic inputs.
Retain attribution and the complete paper citation.
