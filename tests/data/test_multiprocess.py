import multiprocessing as mp
import traceback
from pathlib import Path

import fieldbook as fb


def _writer(root: str, index: int, src: str) -> None:
    import os

    os.environ["FIELDBOOK_STORAGE_ROOT"] = root
    os.environ["FIELDBOOK_PROJECT"] = "test"
    try:
        for i in range(3):
            with fb.dataset("shared").edit(message=f"w{index}-{i}") as edit:
                edit.add(f"w{index}/{i}.txt", src)
    except BaseException:
        Path(root, f"writer-{index}.error").write_text(traceback.format_exc())
        raise


def test_parallel_writers_from_separate_processes(files: Path, storage_root: Path) -> None:
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_writer, args=(str(storage_root), i, str(files / "a" / "1.txt")))
        for i in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    errors = "\n".join(p.read_text() for p in sorted(storage_root.glob("writer-*.error")))
    assert [p.exitcode for p in procs] == [0, 0, 0, 0], errors
    ds = fb.dataset("shared")
    assert ds.index().count_rows() == 12
    assert len(ds.log()) == 12  # one version an edit, whatever order they landed in
