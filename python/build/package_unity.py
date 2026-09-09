"""Create a local UPM .tgz from the Unity package, without models or recordings.

Usage: python -m python.build.package_unity --output-dir C:/some/shared/folder
"""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "unity/Packages/com.domicube.phoneme-matching"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / "package.json").read_text(encoding="utf-8"))
    roots = {"Runtime", "Editor", "Tests", "Samples~", "Documentation~"}
    root_files = {"package.json", "README.md", "CHANGELOG.md", "LICENSE", "LICENSE.md"}
    files = []
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(PACKAGE)
        if path.is_symlink():
            raise ValueError(f"Package must not contain symlinks: {rel}")
        if not path.is_file():
            continue
        root_name = rel.parts[0].removesuffix(".meta")
        if root_name not in roots and root_name not in root_files:
            continue
        if path.suffix.lower() in {".onnx", ".wav", ".mp3", ".mp4", ".pyc"}:
            raise ValueError(f"Unexpected media/model in package: {rel}")
        if not any(part.endswith("~") for part in rel.parts) and path.suffix != ".meta":
            if not path.with_name(path.name + ".meta").exists():
                raise ValueError(f"Missing Unity .meta: {rel}")
        files.append(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{manifest['name']}-{manifest['version']}.tgz"
    if output.resolve().is_relative_to(PACKAGE.resolve()):
        raise ValueError("Write the archive outside the package source directory")
    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                info = archive.gettarinfo(str(path), "package/" + path.relative_to(PACKAGE).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as source:
                    archive.addfile(info, source)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"file": str(output), "version": manifest["version"], "files": len(files),
                      "bytes": output.stat().st_size, "sha256": digest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
