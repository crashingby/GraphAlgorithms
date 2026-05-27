import sys
import os

def convert_konect_to_mtx(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 '{input_file}'")
        return

    edges = []
    max_node = 0
    
    print(f"正在读取并处理: {input_file}...")
    
    with open(input_file, 'r') as f:
        for line in f:
            # 过滤注释行（% 或 # 开头）
            if line.startswith(('%', '#')) or not line.strip():
                continue
            
            try:
                parts = line.split()
                # 获取节点 ID
                u, v = int(parts[0]), int(parts[1])
                
                # 如果原始数据自带权重，则保留；否则设为 1
                w = parts[2] if len(parts) > 2 else "1"
                
                edges.append((u, v, w))
                
                # 追踪最大节点编号以确定矩阵维度
                if u > max_node: max_node = u
                if v > max_node: max_node = v
            except (ValueError, IndexError):
                continue

    with open(output_file, 'w') as f:
        # 1. 写入头部信息
        # 使用 'general' 表示非对称（或由数据自行定义），'integer' 表示权重为整数
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        
        # 2. 写入元数据行: 行数 列数 边数
        f.write(f"{max_node} {max_node} {len(edges)}\n")
        
        # 3. 写入边数据：每行三列 (u v w)
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")

    print(f"--- 转换成功 ---")
    print(f"输出文件: {output_file}")
    print(f"矩阵维度: {max_node} x {max_node}")
    print(f"总边数: {len(edges)}")
    print(f"默认权重已设为 1")

if __name__ == "__main__":
    # 使用方式: python convert.py <输入文件> <输出文件>
    if len(sys.argv) > 2:
        convert_konect_to_mtx(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        out_name = os.path.splitext(sys.argv[1])[0] + ".mtx"
        convert_konect_to_mtx(sys.argv[1], out_name)
    else:
        print("用法: python script.py <input.edges> [output.mtx]")