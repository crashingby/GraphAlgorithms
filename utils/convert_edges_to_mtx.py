import sys
import os

def convert_edges_to_mtx(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 '{input_file}'")
        return

    edges = []
    max_node = 0
    
    print(f"正在读取 {input_file}...")
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '%', '/')):
                continue
            
            try:
                parts = line.split()
                u, v = int(parts[0]), int(parts[1])
                # .mtx 通常从 1 开始计数，如果原数据从 0 开始，此处可改为 u+1, v+1
                w = parts[2] if len(parts) > 2 else "1"
                
                edges.append((u, v, w))
                max_node = max(max_node, u, v)
            except (ValueError, IndexError):
                continue

    with open(output_file, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{max_node} {max_node} {len(edges)}\n")
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")

    print(f"--- 转换成功 ---")
    print(f"输出路径: {os.path.abspath(output_file)}")
    print(f"矩阵规模: {max_node}x{max_node}, 边数: {len(edges)}")

if __name__ == "__main__":
    # 如果用户提供了参数：python script.py in.edges out.mtx
    if len(sys.argv) > 2:
        convert_edges_to_mtx(sys.argv[1], sys.argv[2])
    # 如果用户只提供了一个参数，自动生成输出文件名
    elif len(sys.argv) == 2:
        out_name = os.path.splitext(sys.argv[1])[0] + ".mtx"
        convert_edges_to_mtx(sys.argv[1], out_name)
    # 如果什么都没提供，交互式询问
    else:
        in_file = input("请输入源文件名 (如 data.edges): ").strip()
        out_file = input("请输入输出文件名 (直接回车默认同名 .mtx): ").strip()
        if not out_file:
            out_file = os.path.splitext(in_file)[0] + ".mtx"
        convert_edges_to_mtx(in_file, out_file)