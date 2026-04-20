import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "random_forest.cpp"
    output = root / f"random_forest_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"

    includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"],
        text=True,
    ).strip()

    cmd = [
        "g++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        *shlex.split(includes),
        str(source),
        "-o",
        str(output),
    ]

    print("Compiling random_forest_cpp...")
    subprocess.run(cmd, check=True, cwd=root)
    print(f"Built {output.name}")


if __name__ == "__main__":
    main()
