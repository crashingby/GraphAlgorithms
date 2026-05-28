import sys
from pathlib import Path


def fix_edgelist_to_mtx(input_path, output_path, one_based=True):
    input_path = Path(input_path)
    output_path = Path(output_path)

    edges = []
    max_node = -1
    min_node = 10**30

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("#") or line.startswith("%"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])

            max_node = max(max_node, u, v)
            min_node = min(min_node, u, v)
            edges.append((u, v))

    if not edges:
        raise RuntimeError("没有读到任何边，请检查输入文件")

    if one_based:
        # 原始 edge list 如果是 0-based，需要全部 +1，变成 MatrixMarket 常见的 1-based
        fixed_edges = [(u + 1, v + 1) for u, v in edges]
        n = max_node + 1
    else:
        # 如果你的原始 edge list 已经是 1-based，就不改编号
        fixed_edges = edges
        n = max_node

    with output_path.open("w", encoding="utf-8") as f:
        f.write("%%MatrixMarket matrix coordinate pattern general\n")
        f.write("% generated from edge list\n")
        f.write(f"{n} {n} {len(fixed_edges)}\n")

        for u, v in fixed_edges:
            f.write(f"{u} {v}\n")

    print("done")
    print(f"input      : {input_path}")
    print(f"output     : {output_path}")
    print(f"min_node   : {min_node}")
    print(f"max_node   : {max_node}")
    print(f"nodes      : {n}")
    print(f"edges      : {len(fixed_edges)}")
    print(f"one_based  : {one_based}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage:")
        print("  python3 fix_edgelist_to_mtx.py input.txt output.mtx")
        print()
        print("默认认为原始文件是 0-based，会输出 MatrixMarket 常见的 1-based 编号")
        sys.exit(1)

    fix_edgelist_to_mtx(sys.argv[1], sys.argv[2], one_based=True)