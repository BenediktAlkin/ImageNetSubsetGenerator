from pathlib import Path
import inspect
import shutil

def assert_clean_directory(file_name, subfolder=None):
    test_name = inspect.stack()[1][3]
    if subfolder is not None:
        root = Path(f"tests_output/{file_name}/{test_name}/{subfolder}")
    else:
        root = Path(f"tests_output/{file_name}/{test_name}")
    if root.exists():
        if root.is_dir():
            shutil.rmtree(root)
        else:
            raise NotImplementedError
    assert not root.exists()
    root.mkdir(exist_ok=True, parents=True)
    return root