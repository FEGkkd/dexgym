"""Legacy import path. Implementation lives in hand21_supermimic/supermimic_evaluate.py."""
from ..hand21_supermimic.supermimic_evaluate import (
    summarize_errors as summarize_errors,
    FixedClockHandEnv as FixedClockHandEnv,
    evaluate_subgoals as evaluate_subgoals,
    evaluate_fixed_clock as evaluate_fixed_clock,
    main as main,
)

if __name__ == "__main__":
    main()
