"""Vendored PatchTST reference implementation.

Source: https://github.com/yuqinie98/PatchTST (MIT License)
RevIN source: https://github.com/ts-kim/RevIN (MIT License)

Code is preserved verbatim from the paper authors' repositories except for:
  - Import paths rewritten to relative imports within this subpackage.
  - No functional changes.

PatchTST paper:
  Nie, Nguyen, Sinthong, Kalagnanam (2023), *A Time Series is Worth 64 Words:
  Long-term Forecasting with Transformers*, ICLR 2023.

RevIN paper:
  Kim, Kim, Tae, Park, Choi, Choo (2022), *Reversible Instance Normalization
  for Accurate Time-Series Forecasting against Distribution Shift*, ICLR 2022.
"""

from case_studies.config.patchtst._reference.backbone import PatchTST_backbone

__all__ = ["PatchTST_backbone"]
