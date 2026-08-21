"""
Chinese-SimpleQA profiling / held-out evaluation split utilities.

Formal experiment protocol:
- split by JSON file, never by token/segment;
- stratified by top-level category;
- deterministic seed;
- Profile subset is the only subset allowed to build frequency/coactivation;
- Evaluation subset is the only subset used for final Prefill/Decode metrics.

The manifest stores relative file paths, so it remains usable after moving the
repository.  A lightweight source fingerprint (relative path + size + mtime)
is used to detect dataset changes and regenerate the split when necessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRACE_ROOT = (
    PROJECT_ROOT
    / "deepseek_r1_trace"
    / "cognitivecomputations"
    / "DeepSeek-R1-AWQ"
    / "Chinese-SimpleQA"
)

DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "splits"
    / "chinese_simpleqa_80_20_seed42.json"
)

DEFAULT_PROFILE_CACHE = (
    PROJECT_ROOT
    / "results"
    / "cache"
    / "chinese_simpleqa_profile_80_20_seed42.pkl"
)

TRACE_SPLIT_VERSION = 1
PROFILE_SUBSET = "profile"
EVALUATION_SUBSET = "evaluation"
TRACE_SUBSETS = (PROFILE_SUBSET, EVALUATION_SUBSET)


class TraceSplitError(ValueError):
    pass


def _discover_json_files(trace_root: Path | str) -> tuple[Path, ...]:
    root = Path(trace_root).resolve()
    if not root.exists() or not root.is_dir():
        raise TraceSplitError(f"Trace 根目录不存在或不是目录：{root}")

    files = tuple(
        sorted(
            (path for path in root.rglob("*.json") if path.is_file()),
            key=lambda path: str(path.relative_to(root)),
        )
    )
    if not files:
        raise TraceSplitError(f"Trace 目录下没有 JSON：{root}")
    return files


def _category_of(relative: Path) -> str:
    return relative.parts[0] if len(relative.parts) >= 2 else "__root__"


def fingerprint_trace_files(
    *,
    trace_root: Path | str,
    files: Iterable[Path],
) -> str:
    """Fingerprint a concrete ordered file set without reading file contents."""

    root = Path(trace_root).resolve()
    digest = hashlib.sha256()
    for path in sorted((Path(x).resolve() for x in files), key=lambda p: str(p.relative_to(root))):
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise TraceSplitError(f"文件不位于 Trace 根目录：{path}") from exc
        stat = path.stat()
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_category_seed(seed: int, category: str) -> int:
    raw = f"{seed}:{category}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _split_one_category(
    files: list[str],
    *,
    profile_ratio: float,
    seed: int,
    category: str,
) -> tuple[list[str], list[str]]:
    ordered = sorted(files)
    rng = random.Random(_stable_category_seed(seed, category))
    rng.shuffle(ordered)

    n = len(ordered)
    if n == 1:
        # Tiny synthetic datasets can contain a singleton category.  Keeping it
        # in Profile avoids an empty profile while real Chinese-SimpleQA
        # categories contain many files.
        return ordered, []

    profile_n = int(round(n * profile_ratio))
    profile_n = min(max(profile_n, 1), n - 1)

    profile = sorted(ordered[:profile_n])
    evaluation = sorted(ordered[profile_n:])
    return profile, evaluation


def build_trace_split_manifest(
    *,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    profile_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    if not 0.0 < profile_ratio < 1.0:
        raise TraceSplitError("profile_ratio 必须严格位于 (0, 1)。")

    root = Path(trace_root).resolve()
    files = _discover_json_files(root)

    by_category: dict[str, list[str]] = {}
    for path in files:
        relative = path.relative_to(root)
        category = _category_of(relative)
        by_category.setdefault(category, []).append(str(relative).replace("\\", "/"))

    profile_files: list[str] = []
    evaluation_files: list[str] = []
    category_counts: dict[str, dict[str, int]] = {}

    for category in sorted(by_category):
        profile, evaluation = _split_one_category(
            by_category[category],
            profile_ratio=profile_ratio,
            seed=seed,
            category=category,
        )
        profile_files.extend(profile)
        evaluation_files.extend(evaluation)
        category_counts[category] = {
            "total": len(by_category[category]),
            "profile": len(profile),
            "evaluation": len(evaluation),
        }

    profile_files.sort()
    evaluation_files.sort()

    profile_set = set(profile_files)
    evaluation_set = set(evaluation_files)
    all_rel = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in files
    }

    if profile_set & evaluation_set:
        raise TraceSplitError("Profile / Evaluation 文件发生重叠。")
    if profile_set | evaluation_set != all_rel:
        missing = all_rel - (profile_set | evaluation_set)
        raise TraceSplitError(f"Split 未覆盖全部文件：missing={len(missing)}")
    if not evaluation_files:
        raise TraceSplitError("Evaluation subset 为空，无法做 held-out 评估。")

    source_fingerprint = fingerprint_trace_files(trace_root=root, files=files)
    profile_paths = [root / rel for rel in profile_files]
    evaluation_paths = [root / rel for rel in evaluation_files]

    return {
        "split_version": TRACE_SPLIT_VERSION,
        "dataset": "Chinese-SimpleQA",
        "profile_ratio": float(profile_ratio),
        "evaluation_ratio": float(1.0 - profile_ratio),
        "seed": int(seed),
        "source_file_count": len(files),
        "source_fingerprint": source_fingerprint,
        "category_counts": category_counts,
        "profile": {
            "file_count": len(profile_files),
            "fingerprint": fingerprint_trace_files(
                trace_root=root,
                files=profile_paths,
            ),
            "files": profile_files,
        },
        "evaluation": {
            "file_count": len(evaluation_files),
            "fingerprint": fingerprint_trace_files(
                trace_root=root,
                files=evaluation_paths,
            ),
            "files": evaluation_files,
        },
    }


def save_trace_split_manifest(*, manifest: dict, output_path: Path | str) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def load_trace_split_manifest(path: Path | str) -> dict:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise TraceSplitError(f"Split manifest 不存在：{file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceSplitError(f"无法读取 split manifest：{file_path}") from exc

    if not isinstance(data, dict):
        raise TraceSplitError("Split manifest 最外层必须是 dict。")
    if data.get("split_version") != TRACE_SPLIT_VERSION:
        raise TraceSplitError(
            f"Split manifest version 不匹配：{data.get('split_version')}"
        )
    for subset in TRACE_SUBSETS:
        section = data.get(subset)
        if not isinstance(section, dict) or not isinstance(section.get("files"), list):
            raise TraceSplitError(f"Split manifest 缺少 {subset}.files。")
    return data


def ensure_trace_split(
    *,
    trace_root: Path | str = DEFAULT_TRACE_ROOT,
    manifest_path: Path | str = DEFAULT_SPLIT_MANIFEST,
    profile_ratio: float = 0.8,
    seed: int = 42,
    force: bool = False,
    verbose: bool = True,
) -> tuple[dict, Path, bool]:
    """Return (manifest, path, rebuilt)."""

    root = Path(trace_root).resolve()
    output = Path(manifest_path).resolve()
    current_files = _discover_json_files(root)
    current_source_fingerprint = fingerprint_trace_files(
        trace_root=root,
        files=current_files,
    )

    if output.exists() and not force:
        try:
            existing = load_trace_split_manifest(output)
            same_protocol = (
                float(existing.get("profile_ratio", -1.0)) == float(profile_ratio)
                and int(existing.get("seed", -1)) == int(seed)
            )
            same_source = existing.get("source_fingerprint") == current_source_fingerprint
            if same_protocol and same_source:
                if verbose:
                    print(
                        "[TraceSplit] reuse "
                        f"profile={existing['profile']['file_count']}, "
                        f"evaluation={existing['evaluation']['file_count']}, "
                        f"seed={seed}"
                    )
                return existing, output, False
        except TraceSplitError:
            pass

    manifest = build_trace_split_manifest(
        trace_root=root,
        profile_ratio=profile_ratio,
        seed=seed,
    )
    saved = save_trace_split_manifest(manifest=manifest, output_path=output)
    if verbose:
        print(
            "[TraceSplit] built "
            f"profile={manifest['profile']['file_count']}, "
            f"evaluation={manifest['evaluation']['file_count']}, "
            f"seed={seed} -> {saved}"
        )
    return manifest, saved, True


def resolve_trace_files(
    *,
    trace_root: Path | str,
    manifest_path: Path | str | None = None,
    subset: str = EVALUATION_SUBSET,
) -> tuple[Path, ...]:
    root = Path(trace_root).resolve()
    if manifest_path is None:
        return _discover_json_files(root)

    if subset not in TRACE_SUBSETS:
        raise TraceSplitError(f"非法 subset={subset!r}，允许值={TRACE_SUBSETS}")

    manifest = load_trace_split_manifest(manifest_path)
    section = manifest[subset]
    relative_files = section["files"]
    files: list[Path] = []
    for relative in relative_files:
        if not isinstance(relative, str):
            raise TraceSplitError(f"{subset}.files 中存在非字符串路径。")
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TraceSplitError(f"Manifest 路径越出 Trace 根目录：{relative}") from exc
        if not path.exists() or not path.is_file():
            raise TraceSplitError(f"Manifest 引用文件不存在：{path}")
        files.append(path)

    if not files:
        raise TraceSplitError(f"{subset} subset 为空。")
    return tuple(files)


def manifest_protocol_summary(
    *,
    manifest_path: Path | str,
    profile_subset: str = PROFILE_SUBSET,
    evaluation_subset: str = EVALUATION_SUBSET,
) -> dict:
    manifest = load_trace_split_manifest(manifest_path)
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "dataset": manifest.get("dataset", "Chinese-SimpleQA"),
        "split_version": manifest.get("split_version"),
        "seed": manifest.get("seed"),
        "profile_ratio": manifest.get("profile_ratio"),
        "source_file_count": manifest.get("source_file_count"),
        "source_fingerprint": manifest.get("source_fingerprint"),
        "profile_subset": profile_subset,
        "profile_file_count": manifest[profile_subset]["file_count"],
        "profile_fingerprint": manifest[profile_subset]["fingerprint"],
        "evaluation_subset": evaluation_subset,
        "evaluation_file_count": manifest[evaluation_subset]["file_count"],
        "evaluation_fingerprint": manifest[evaluation_subset]["fingerprint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按类别分层生成 Chinese-SimpleQA Profile/Held-out 文件划分。"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--profile-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest, saved, rebuilt = ensure_trace_split(
        trace_root=args.root,
        manifest_path=args.output,
        profile_ratio=args.profile_ratio,
        seed=args.seed,
        force=args.force,
        verbose=True,
    )
    print("\n========== Trace Split ==========")
    print(f"Manifest：{saved}")
    print(f"Rebuilt：{rebuilt}")
    print(f"Total：{manifest['source_file_count']}")
    print(f"Profile：{manifest['profile']['file_count']}")
    print(f"Evaluation：{manifest['evaluation']['file_count']}")
    print(f"Seed：{manifest['seed']}")
    print(f"Profile Ratio：{manifest['profile_ratio']:.4f}")


if __name__ == "__main__":
    main()
