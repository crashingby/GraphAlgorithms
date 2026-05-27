#include <math.h>
#include <stdio.h>
#include <algorithm>
/**
 * 存储以坐标（COO）格式表示的边。COO格式将每条边表示为元组（行、列、值）
*/
typedef struct CooEdgeTuple {
	int row;
	int col;
	float val;

	CooEdgeTuple(int row, int col, float val) : row(row), col(col), val(val) {}

	void Val(float &value)
	{
		val = value;
	}
}CooEdgeTuple;

/**
 * Comparator for sorting COO sparse format edges
 */
bool TupleCompare (CooEdgeTuple elem1,CooEdgeTuple elem2)
{
	if (elem1.row < elem2.row) {
		return true;
	} 
	return false;
}
bool TupleCompare2 (   CooEdgeTuple elem1,CooEdgeTuple elem2)
{
	if (elem1.col < elem2.col) {
	    return true;
	} 
	return false;
}

/**
 * 表示CSR（Compressed Sparse Row）稀疏格式的图数据结构
 */
typedef struct CsrGraph
{
    int nodes;             //节点数量
    int edges;             //边数量
    
    int *row_offsets;      //存储行偏移值的数组，描述每个节点的出边（边）邻居节点在column_indices数组中的起始位置。
    int *column_indices;   //存储列索引的数组，用于描述每个边的目标节点。
    
    int *column_offsets;   //存储列偏移值的数组。有向图中描述每个节点的入边的邻居节点在row_indices的起始位置
    int *row_indices;      //存储行索引的数组。有向图中描述每个入边的源节点。
    
    int *edge_values;    //存储边的权重或属性值的数组。


    /**
     * Constructor
     */
    CsrGraph()
    {
      nodes = 0;
      edges = 0;
      row_offsets = NULL;
      column_indices = NULL;
      edge_values = NULL;
    }

    void FromCoo(CooEdgeTuple *coo, int coo_nodes, int coo_edges, int undirected)
    {
     // printf("Converting %d vertices, %d directed edges to CSR format... ", coo_nodes, coo_edges);
      
      //填充csr对象
      this->nodes = coo_nodes;
      this->edges = coo_edges;

      row_offsets = (int*) malloc(sizeof(int) * (coo_nodes + 1));
      column_indices = (int*) malloc(sizeof(int) * coo_edges);
      if (!undirected)//有向图存储入边
        {
          column_offsets = (int*) malloc(sizeof(int) * (coo_nodes + 1));
          row_indices = (int*) malloc(sizeof(int) * coo_edges);
        }

      edge_values = (int*) malloc(sizeof(int) * coo_edges) ;
    
        //coo按行升序排序
        std::stable_sort(coo, coo + coo_edges, TupleCompare);

        int prev_row = -1;
        for (int edge = 0; edge < coo_edges; edge++) //edge是行偏移
        {

          int current_row = coo[edge].row;
          // 填充截止到当前行
          for (int row = prev_row + 1; row <= current_row; row++)
          {
            row_offsets[row] = edge;
          }
          prev_row = current_row;

          column_indices[edge] = coo[edge].col;
    
          edge_values[edge] = coo[edge].val;
//            coo[edge].Val(edge_values[edge]);
        }

        // Fill out any trailing edgeless nodes (and the end-of-list element)
        for (int row = prev_row + 1; row <= nodes; row++)
        {
          row_offsets[row] = edges;
        }


        if (!undirected)//有向图
        {
          // Sort COO by col
          std::stable_sort(coo, coo + coo_edges, TupleCompare2);

          int prev_col = -1;
          for (int edge = 0; edge < edges; edge++)
          {

            int current_col = coo[edge].col;
            // Fill in rows up to and including the current row
            for (int col = prev_col + 1; col <= current_col; col++)
            {
              column_offsets[col] = edge;
            }
            prev_col = current_col;

            row_indices[edge] = coo[edge].row;
          }
          // Fill out any trailing edgeless nodes (and the end-of-list element)
          for (int col = prev_col + 1; col <= nodes; col++)
          {
            column_offsets[col] = edges;
          }
        }
      

    }
   
       
  
}CsrGraph;


/**
 * 打印CSR图数据
*/
void printCSR(CsrGraph &csr_graph){
    printf("crs_graph:nodes %d, edges  %d  \n",csr_graph.nodes,csr_graph.edges);
    printf("crs_graph:row_offsets : ");
    for(int i=0;i<csr_graph.nodes+1;i++){
        printf("%d  ",csr_graph.row_offsets[i]);
    }
    printf("\ncrs_graph:column_indices : ");
    for(int i=0;i<csr_graph.edges;i++){
        printf("%d  ",csr_graph.column_indices[i]);
    }
    if(csr_graph.column_offsets != NULL){
    printf("\ncrs_graph:column_offsets : ");
    for(int i=0;i<csr_graph.nodes+1;i++){
    printf("%d  ",csr_graph.column_offsets[i]);
    }
    }
    if(csr_graph.row_indices != NULL){
    printf("\ncrs_graph:row_indices : ");
    for(int i=0;i<csr_graph.edges;i++){
    printf("%d  ",csr_graph.row_indices[i]);
    } 
    }
    printf("\n");     
    }
 





/**
 * 从图文件中读取MARKET格式的图形数据并将其转换为压缩稀疏行（CSR）格式。
 */
int BuildMarketGraph(const char* graph_filename, CsrGraph &csr_graph, bool undirected)
{ 
    // Read from file
    FILE *f_in = fopen(graph_filename, "r");
    char line[1024];
    int edges_read = -1;//已读取的边
    int nodes = 0;
    int edges = 0;
    CooEdgeTuple *coo = NULL;
    if (f_in) {
        // printf("Reading from %s:\n", graph_filename);
        //循环读取图文件
        while(true) {
            //从文件流 f_in 中读取一行文本，并将其存储在字符串 line 中。然后它检查是否成功读取行
            if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
               break; 
            }
            if (line[0] == '%') {
              // 注释，忽略
            }else if(edges_read == -1){   //读取第一行：表示图的维度（行和列） 边数

                long long ll_nodes_x, ll_nodes_y, ll_edges;//存储从行中提取的值
                sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges);
                nodes = ll_nodes_x;
                edges = (undirected) ? ll_edges * 2 : ll_edges;//如果图是无向的，还会插入反向边。
                
                // Allocate coo graph
                coo = (CooEdgeTuple*) malloc(sizeof(CooEdgeTuple) * edges);
                edges_read++;
            }else{  //读取边
                long long ll_row, ll_col;
                double edge_value = 1; // 如果边描述行中提供了具体的值，那么它将覆盖这个默认值。
                int nread = sscanf(line, "%lld %lld %lf", &ll_row, &ll_col, &edge_value);
           
                coo[edges_read].row = ll_row ;	// zero-based array
                coo[edges_read].col = ll_col ;	// zero-based array
                coo[edges_read].val =(float) edge_value;

                edges_read++;

                if (undirected) {  //图是无向的，插入反向边。
                  coo[edges_read].row = ll_col ;	// zero-based array
                  coo[edges_read].col = ll_row ;	// zero-based array
                  coo[edges_read].val = (float)edge_value;
                  edges_read++;
			          }
            }
        }
      
        // Convert COO to CSR
        csr_graph.FromCoo(coo,nodes,edges,undirected);
        free(coo);
	      fflush(stdout);
        return 0;
    } else {
        perror("Unable to open file");
        return -1;
    }

}


// ============================================================
// 按顶点范围切图
// ============================================================
void SplitGraphByVertex(
    const CsrGraph &full_graph,
    CsrGraph &subgraph,
    int start_node,
    int end_node)
{
    int sub_nodes = end_node - start_node;

    std::vector<CooEdgeTuple> sub_edges;

    // 遍历属于当前 partition 的顶点
    for (int u = start_node; u < end_node; u++) {

        int row_start = full_graph.row_offsets[u];
        int row_end   = full_graph.row_offsets[u + 1];

        for (int e = row_start; e < row_end; e++) {

            int v = full_graph.column_indices[e];

            // 顶点重新映射到局部编号
            int local_u = u - start_node;

            // 注意：
            // 这里 v 保持全局 ID
            // 因为 BFS pull 需要访问全局 value
            sub_edges.emplace_back(local_u, v, 1.0f);
        }
    }

    subgraph.FromCoo(
        sub_edges.data(),
        sub_nodes,
        sub_edges.size(),
        false);

    printf("Subgraph [%d, %d) nodes=%d edges=%zu\n",
           start_node,
           end_node,
           sub_nodes,
           sub_edges.size());
}
