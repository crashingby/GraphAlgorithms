#!/usr/bin/env python3
"""
Download, extract, convert, and summarize benchmark graph datasets.

Outputs:
  - archives/extracted files under temp_downloads/
  - converted MatrixMarket files under dataset/<name>.mtx
  - compact vertex maps under temp_downloads/mappings/<name>.map.tsv
  - optional stats markdown under temp_downloads/dataset_stats_summary.md

The converted MatrixMarket files use 0-based compact vertex ids because the
CUDA kernels in this project expect array-friendly vertex ids.
"""

from __future__ import annotations

import argparse
import gzip
import html.parser
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMP = ROOT / "temp_downloads"
DEFAULT_DATASET_DIR = ROOT / "dataset"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    page_url: str
    direct_url: str | None = None


DATASETS: list[DatasetSpec] = [
    DatasetSpec("cit-HepPh", "https://snap.stanford.edu/data/cit-HepPh.html", "https://snap.stanford.edu/data/cit-HepPh.txt.gz"),
    DatasetSpec("soc-Epinions1", "https://snap.stanford.edu/data/soc-Epinions1.html", "https://snap.stanford.edu/data/soc-Epinions1.txt.gz"),
    DatasetSpec("soc-Slashdot0811", "https://snap.stanford.edu/data/soc-Slashdot0811.html", "https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz"),
    DatasetSpec("ia-enron-email-dynamic", "https://networkrepository.com/ia-enron-email-dynamic.php"),
    DatasetSpec("tech-RL-caida", "https://networkrepository.com/tech-RL-caida.php"),
    DatasetSpec("web-NotreDame", "https://snap.stanford.edu/data/web-NotreDame.html", "https://snap.stanford.edu/data/web-NotreDame.txt.gz"),
    DatasetSpec("soc-themarker", "https://networkrepository.com/soc-themarker.php"),
    DatasetSpec("socfb-Harvard1", "https://networkrepository.com/socfb-Harvard1.php"),
    DatasetSpec("socfb-Oklahoma97", "https://networkrepository.com/socfb-Oklahoma97.php"),
    DatasetSpec("web-Stanford", "https://snap.stanford.edu/data/web-Stanford.html", "https://snap.stanford.edu/data/web-Stanford.txt.gz"),
    DatasetSpec("socfb-Indiana69", "https://networkrepository.com/socfb-Indiana69.php"),
    DatasetSpec("socfb-Penn94", "https://networkrepository.com/socfb-Penn94.php"),
    DatasetSpec("socfb-UF21", "https://networkrepository.com/socfb-UF21.php"),
    DatasetSpec("socfb-Texas84", "https://networkrepository.com/socfb-Texas84.php"),
    DatasetSpec("soc-BlogCatalog", "https://networkrepository.com/soc-BlogCatalog.php"),
    DatasetSpec("soc-LiveMocha", "https://networkrepository.com/soc-LiveMocha.php"),
    DatasetSpec("web-Google", "https://snap.stanford.edu/data/web-Google.html", "https://snap.stanford.edu/data/web-Google.txt.gz"),
    DatasetSpec("soc-buzznet", "https://networkrepository.com/soc-buzznet.php"),
    DatasetSpec("roadNet-CA", "https://snap.stanford.edu/data/roadNet-CA.html", "https://snap.stanford.edu/data/roadNet-CA.txt.gz"),
    DatasetSpec("soc-flickr", "https://networkrepository.com/soc-flickr.php"),
    DatasetSpec("soc-catster", "https://networkrepository.com/soc-catster.php"),
    DatasetSpec("soc-LiveJournal1", "https://snap.stanford.edu/data/soc-LiveJournal1.html", "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"),
]


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_text(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "GraphAlgorithm-dataset-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def discover_download_url(spec: DatasetSpec, timeout: int) -> str:
    if spec.direct_url:
        return spec.direct_url

    html = fetch_text(spec.page_url, timeout)
    parser = LinkParser()
    parser.feed(html)

    candidates: list[str] = []
    for href in parser.links:
        absolute = urllib.parse.urljoin(spec.page_url, href)
        lower = absolute.lower()
        if lower.endswith((".zip", ".7z", ".gz", ".bz2")):
            candidates.append(absolute)

    # Prefer the easy-to-extract ZIP exposed by NetworkRepository/nrvis.
    for url in candidates:
        if url.lower().endswith(".zip") and "nrvis.com" in url.lower():
            return url
    for url in candidates:
        if url.lower().endswith(".zip"):
            return url
    for url in candidates:
        if url.lower().endswith(".gz"):
            return url
    if candidates:
        return candidates[0]

    # NetworkRepository pages usually follow this stable nrvis pattern.
    slug = spec.name
    return f"https://nrvis.com/download/data/{slug[:3]}/{slug}.zip"


def download_file(url: str, dest: Path, timeout: int, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log(f"[skip] archive exists: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "GraphAlgorithm-dataset-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp.replace(dest)


def extract_archive(archive: Path, extract_dir: Path, force: bool) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if marker.exists() and not force:
        return list(extract_dir.rglob("*"))

    if force and extract_dir.exists():
        for p in extract_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    lower = archive.name.lower()
    log(f"[extract] {archive}")
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    elif lower.endswith(".tar.gz") or lower.endswith(".tgz") or lower.endswith(".tar"):
        with tarfile.open(archive) as tf:
            tf.extractall(extract_dir)
    elif lower.endswith(".gz"):
        out_name = archive.name[:-3]
        out_path = extract_dir / out_name
        with gzip.open(archive, "rb") as src, out_path.open("wb") as out:
            shutil.copyfileobj(src, out)
    else:
        shutil.copy2(archive, extract_dir / archive.name)

    marker.write_text("ok\n")
    return list(extract_dir.rglob("*"))


def candidate_graph_files(extract_dir: Path) -> list[Path]:
    suffix_priority = {
        ".mtx": 0,
        ".edges": 1,
        ".txt": 2,
        ".csv": 3,
        ".tsv": 3,
        ".out": 4,
        ".dat": 5,
    }
    files = [
        p for p in extract_dir.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in suffix_priority
        and "readme" not in p.name.lower()
        and "meta" not in p.name.lower()
    ]
    files.sort(key=lambda p: (suffix_priority.get(p.suffix.lower(), 99), -p.stat().st_size))
    return files


def parse_int_pair(parts: list[str]) -> tuple[int, int] | None:
    ints: list[int] = []
    for part in parts:
        token = part.strip().strip(",")
        if not token:
            continue
        try:
            ints.append(int(token))
        except ValueError:
            continue
        if len(ints) == 2:
            return ints[0], ints[1]
    return None


def read_edges_from_mtx(path: Path) -> tuple[list[tuple[int, int]], int | None, bool]:
    edges: list[tuple[int, int]] = []
    header_n: int | None = None
    saw_size = False
    one_based = False
    min_id: int | None = None
    max_id: int | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.replace(",", " ").split()
            if not saw_size and len(parts) >= 3:
                try:
                    rows, cols, _ = map(int, parts[:3])
                    header_n = max(rows, cols)
                    saw_size = True
                    continue
                except ValueError:
                    pass
            pair = parse_int_pair(parts)
            if pair is None:
                continue
            u, v = pair
            edges.append((u, v))
            min_id = u if min_id is None else min(min_id, u, v)
            max_id = u if max_id is None else max(max_id, u, v)

    if edges and header_n is not None and min_id is not None and max_id is not None:
        one_based = min_id >= 1 and max_id <= header_n
    return edges, header_n, one_based


def read_edges_from_edgelist(path: Path) -> tuple[list[tuple[int, int]], int | None, bool]:
    edges: list[tuple[int, int]] = []
    declared_nodes: int | None = None
    min_id: int | None = None

    node_re = re.compile(r"(?:nodes|vertices)\s*[:=]\s*(\d+)", re.IGNORECASE)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("#", "%", "//")):
                m = node_re.search(s)
                if m:
                    declared_nodes = int(m.group(1))
                continue

            parts = s.replace(",", " ").split()
            pair = parse_int_pair(parts)
            if pair is None:
                continue
            u, v = pair
            edges.append((u, v))
            min_id = u if min_id is None else min(min_id, u, v)

    # SNAP is usually 0-based or arbitrary labels; NetworkRepository often ships
    # 1-based MatrixMarket. For plain edge lists, compact relabeling makes this
    # distinction irrelevant, so keep ids as labels and remap below.
    return edges, declared_nodes, False


def read_raw_edges(path: Path) -> tuple[list[tuple[int, int]], int | None, bool]:
    if path.suffix.lower() == ".mtx":
        return read_edges_from_mtx(path)
    return read_edges_from_edgelist(path)


def write_compact_mtx(
    raw_edges: Iterable[tuple[int, int]],
    output_path: Path,
    map_path: Path,
    header_nodes: int | None,
    one_based: bool,
    dedupe: bool,
) -> tuple[int, int]:
    mapping: dict[int, int] = {}
    compact_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def get_id(label: int) -> int:
        mapped = mapping.get(label)
        if mapped is not None:
            return mapped
        mapped = len(mapping)
        mapping[label] = mapped
        return mapped

    for u, v in raw_edges:
        if one_based:
            u -= 1
            v -= 1
        if u == v or u < 0 or v < 0:
            continue
        cu, cv = get_id(u), get_id(v)
        edge = (cu, cv)
        if dedupe and edge in seen:
            continue
        seen.add(edge)
        compact_edges.append(edge)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("%%MatrixMarket matrix coordinate pattern general\n")
        f.write(f"{len(mapping)} {len(mapping)} {len(compact_edges)}\n")
        for u, v in compact_edges:
            f.write(f"{u} {v}\n")

    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", encoding="utf-8") as f:
        f.write("original_id\tcompact_id\n")
        for original, compact in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{original}\t{compact}\n")

    if header_nodes is not None and header_nodes > len(mapping):
        log(f"[note] header nodes={header_nodes}, observed/relabelled nodes={len(mapping)}")
    return len(mapping), len(compact_edges)


def iter_edge_pairs(path: Path) -> Iterable[tuple[int, int]]:
    saw_mtx_size = False
    is_mtx = path.suffix.lower() == ".mtx"
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("%", "#", "//")):
                continue
            parts = s.replace(",", " ").split()
            if is_mtx and not saw_mtx_size:
                # MatrixMarket size line: rows cols nnz
                if len(parts) >= 3:
                    try:
                        int(parts[0]); int(parts[1]); int(parts[2])
                        saw_mtx_size = True
                        continue
                    except ValueError:
                        pass
            pair = parse_int_pair(parts)
            if pair is not None:
                yield pair


def convert_streaming_compact_mtx(
    source: Path,
    output_path: Path,
    map_path: Path,
    dedupe: bool,
) -> tuple[int, int]:
    mapping: dict[int, int] = {}
    seen: set[tuple[int, int]] | None = set() if dedupe else None
    edge_count = 0

    def get_id(label: int) -> int:
        mapped = mapping.get(label)
        if mapped is not None:
            return mapped
        mapped = len(mapping)
        mapping[label] = mapped
        return mapped

    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_path = output_path.with_suffix(output_path.suffix + ".body.tmp")
    final_tmp = output_path.with_suffix(output_path.suffix + ".tmp")

    with body_path.open("w", encoding="utf-8") as body:
        for u, v in iter_edge_pairs(source):
            if u == v or u < 0 or v < 0:
                continue
            cu = get_id(u)
            cv = get_id(v)
            edge = (cu, cv)
            if seen is not None:
                if edge in seen:
                    continue
                seen.add(edge)
            body.write(f"{cu} {cv}\n")
            edge_count += 1

    with final_tmp.open("w", encoding="utf-8") as out:
        out.write("%%MatrixMarket matrix coordinate pattern general\n")
        out.write(f"{len(mapping)} {len(mapping)} {edge_count}\n")
        with body_path.open("r", encoding="utf-8") as body:
            shutil.copyfileobj(body, out, length=1024 * 1024)
    final_tmp.replace(output_path)
    body_path.unlink(missing_ok=True)

    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", encoding="utf-8") as f:
        f.write("original_id\tcompact_id\n")
        for original, compact in mapping.items():
            f.write(f"{original}\t{compact}\n")

    return len(mapping), edge_count


def convert_dataset(
    spec: DatasetSpec,
    extract_dir: Path,
    dataset_dir: Path,
    temp_dir: Path,
    force: bool,
    dedupe: bool,
) -> Path:
    out_path = dataset_dir / f"{spec.name}.mtx"
    map_path = temp_dir / "mappings" / f"{spec.name}.map.tsv"
    if out_path.exists() and not force:
        log(f"[skip] converted exists: {out_path}")
        return out_path

    candidates = candidate_graph_files(extract_dir)
    if not candidates:
        raise RuntimeError(f"no graph-like file found under {extract_dir}")

    source = candidates[0]
    log(f"[convert] {spec.name}: {source} -> {out_path}")
    n, m = convert_streaming_compact_mtx(source, out_path, map_path, dedupe)
    log(f"[ok] {spec.name}: nodes={n}, edges={m}, map={map_path}")
    return out_path

def run_stats(dataset_names: list[str], output_md: Path, timeout: int | None) -> None:
    rows: list[dict[str, str]] = []
    metric_names = [
        "|V|",
        "|E|",
        "dmax",
        "davg",
        "r",
        "|T|",
        "|T|avg",
        "|T|max",
        "κavg",
        "κ",
        "Kmax",
        "ωlb",
    ]

    for name in dataset_names:
        cmd = [sys.executable, str(ROOT / "scripts" / "utils" / "dataset_stats.py"), name]
        log(f"[stats] {name}")
        row = {"dataset": name}
        try:
            cp = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout if timeout and timeout > 0 else None,
                check=True,
            )
            for line in cp.stdout.splitlines():
                parts = line.split()
                if not parts:
                    continue
                metric = parts[0]
                if metric in metric_names:
                    row[metric] = parts[-1]
        except subprocess.TimeoutExpired:
            row["error"] = f"timeout>{timeout}s"
        except subprocess.CalledProcessError as exc:
            row["error"] = exc.stderr.strip().splitlines()[-1] if exc.stderr.strip() else "failed"
        rows.append(row)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    headers = ["dataset"] + metric_names + ["error"]
    def md_cell(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    with output_md.open("w", encoding="utf-8") as f:
        f.write("# Dataset Statistics Summary\n\n")
        f.write("| " + " | ".join(md_cell(h) for h in headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(md_cell(row.get(h, "")) for h in headers) + " |\n")
    log(f"[stats] wrote {output_md}")


def selected_specs(names: list[str] | None) -> list[DatasetSpec]:
    if not names:
        return DATASETS
    wanted = set(names)
    by_name = {d.name: d for d in DATASETS}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise SystemExit(f"unknown dataset(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def archive_name_from_url(url: str, dataset_name: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    if not name:
        name = dataset_name + ".download"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert benchmark graph datasets.")
    parser.add_argument("datasets", nargs="*", help="dataset names to process; default: all")
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="redownload/reextract/reconvert")
    parser.add_argument("--no-download", action="store_true", help="only convert existing archives/extracted files")
    parser.add_argument("--no-stats", action="store_true", help="skip dataset_stats.py summary")
    parser.add_argument("--stats-timeout", type=int, default=0, help="per-dataset stats timeout in seconds; 0 means no timeout")
    parser.add_argument("--dedupe", action="store_true", help="deduplicate directed edges during conversion; slower and memory-heavy for very large graphs")
    parser.add_argument("--list", action="store_true", help="print known dataset names and exit")
    args = parser.parse_args()

    if args.list:
        for spec in DATASETS:
            print(spec.name)
        return

    specs = selected_specs(args.datasets)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    (args.temp_dir / "archives").mkdir(parents=True, exist_ok=True)
    (args.temp_dir / "extracted").mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    converted: list[str] = []
    for spec in specs:
        log(f"\n=== {spec.name} ===")
        url = discover_download_url(spec, args.timeout)
        archive = args.temp_dir / "archives" / archive_name_from_url(url, spec.name)
        extract_dir = args.temp_dir / "extracted" / spec.name

        if not args.no_download:
            download_file(url, archive, args.timeout, args.force)
        if not archive.exists():
            raise RuntimeError(f"archive missing for {spec.name}: {archive}")

        extract_archive(archive, extract_dir, args.force)
        convert_dataset(
            spec,
            extract_dir,
            args.dataset_dir,
            args.temp_dir,
            args.force,
            dedupe=args.dedupe,
        )
        converted.append(spec.name)

    if not args.no_stats:
        run_stats(
            converted,
            args.temp_dir / "dataset_stats_summary.md",
            args.stats_timeout if args.stats_timeout > 0 else None,
        )


if __name__ == "__main__":
    main()
