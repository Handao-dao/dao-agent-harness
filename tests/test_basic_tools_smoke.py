from __future__ import annotations

from agent_harness.tools import BashTool, EditTool, FindTool, GrepTool, WriteTool


async def test_basic_coding_tools_smoke(tmp_path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("value = 'old'\n", encoding="utf-8")

    found = await FindTool(tmp_path).execute({"pattern": "*.py"})
    searched = await GrepTool(tmp_path).execute({"pattern": "old", "path": "src", "glob": "*.py"})
    await WriteTool(tmp_path).execute({"path": "notes.txt", "content": "created"})
    await EditTool(tmp_path).execute(
        {
            "path": "src/example.py",
            "edits": [{"oldText": "'old'", "newText": "'new'"}],
        }
    )
    shell = await BashTool(tmp_path).execute({"command": "echo dao"})

    assert "src/example.py" in found.content
    assert "src/example.py:1:" in searched.content
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "created"
    assert source.read_text(encoding="utf-8") == "value = 'new'\n"
    assert "dao" in shell.content
