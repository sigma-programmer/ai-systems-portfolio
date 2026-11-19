"""Demo: cost model, diagrams, ADR validation and a site build (no benchmarks run)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "common"))

from teardown.__main__ import main  # noqa: E402

print("=== Cost per task ===")
main(["cost"])
print("\n=== Platform topology (ASCII) ===")
main(["diagrams", "--format", "ascii"])
print("\n=== ADR validation ===")
main(["adrs"])
with tempfile.TemporaryDirectory() as d:
    print("\n=== Site build ===")
    main(["site", "--out", d])
print("\nRun `python -m teardown bench --repeats 2` to execute all project benchmarks.")
