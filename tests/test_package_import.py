from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_package_import_does_not_eagerly_load_components() -> None:
    project_root = Path(__file__).parents[1]
    script = (
        "import json, sys; "
        "sys.path.insert(0, 'src'); "
        "import agent_harness; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.startswith('agent_harness.'))))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_production_subpackages_do_not_export_test_doubles() -> None:
    from agent_harness import providers, tools

    assert not hasattr(providers, "ScriptedProvider")
    assert not hasattr(providers, "ModelRequest")
    assert not hasattr(tools, "FakeTool")
