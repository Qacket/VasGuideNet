import numpy as np
import maxflow
from skimage import morphology
from scipy import ndimage

def mrf_probability_refinement(prob_map, lambda_param=0.5, connectivity=6):
    """
    基于概率图的MRF优化处理
    Parameters:
        prob_map (np.ndarray): 3D概率图 [0,1]
        lambda_param (float): 平滑项权重 (建议0.3-0.8)
        connectivity (int): 邻域连接方式 (6/26邻域)
    Returns:
        np.ndarray: 优化后的二值分割图
    """
    # 参数校验
    assert prob_map.ndim == 3, "输入必须为3D数组"
    assert 0 <= prob_map.min() and prob_map.max() <= 1, "概率值需在[0,1]范围内"
    
    # 初始化图结构
    height, width, depth = prob_map.shape
    graph = maxflow.Graph[float]()
    nodeids = graph.add_grid_nodes(prob_map.shape)
    
    # 构建数据项 (一元势能)
    # 使用负对数似然作为数据代价
    prob_map = np.clip(prob_map, 1e-6, 1-1e-6)  # 防止数值不稳定
    data_cost_1 = -np.log(prob_map)
    data_cost_0 = -np.log(1 - prob_map)
    
    # 构建邻接边 (二元势能)
    if connectivity == 6:
        # 6邻域连接 (共享面)
        structure = np.zeros((3,3,3), dtype=bool)
        structure[1,1,:] = True
        structure[1,:,1] = True
        structure[:,1,1] = True
    else:  # 26邻域
        structure = np.ones((3,3,3), dtype=bool)
    
    # 添加相邻边
    for axis in range(3):
        for direction in [-1, 1]:
            edge_weights = lambda_param * np.exp(-(np.roll(prob_map, direction, axis) - prob_map)**2)
            graph.add_grid_edges(nodeids, 
                               weights=edge_weights,
                               structure=np.roll(structure, direction, axis),
                               symmetric=True)
    
    # 添加终端边 (数据项)
    graph.add_grid_tedges(nodeids, data_cost_0, data_cost_1)
    
    # 执行图切割优化
    graph.maxflow()
    return graph.get_grid_segments(nodeids).astype(np.uint8)

def morphology_postprocess(binary_volume, min_size=8, close_r=2):
    """
    三维形态学后处理
    Parameters:
        binary_volume (np.ndarray): 二值分割结果
        min_size (int): 最小保留体积 (体素数)
        close_r (int): 闭运算结构球半径(mm)
    Returns:
        np.ndarray: 优化后的二值分割图
    """
    # 三维闭运算填充小孔
    struct_elem = morphology.ball(close_r)
    closed = morphology.binary_closing(binary_volume, struct_elem)
    
    # 连通域分析去除小噪声
    labeled_volume, num_labels = ndimage.label(closed)
    component_sizes = np.bincount(labeled_volume.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_volume]
    cleaned = closed.copy()
    cleaned[too_small_mask] = 0
    
    # 骨架修剪 (移除短分支)
    skeleton = morphology.skeletonize_3d(cleaned)
    pruned = _prune_skeleton(skeleton, min_length=3)
    
    return pruned

def _prune_skeleton(skeleton, min_length=3):
    """三维骨架修剪辅助函数"""
    # 端点检测
    endpoints = np.zeros_like(skeleton)
    for z in range(1, skeleton.shape[0]-1):
        for y in range(1, skeleton.shape[1]-1):
            for x in range(1, skeleton.shape[2]-1):
                if skeleton[z,y,x]:
                    neighborhood = skeleton[z-1:z+2, y-1:y+2, x-1:x+2]
                    if np.sum(neighborhood) <= 2:
                        endpoints[z,y,x] = 1
    # 移除短分支
    return skeleton & ~endpoints  # 简化实现，完整实现需要跟踪分支长度

def full_postprocessing(prob_volume):
    # 配置处理参数
    pipeline = {
        'mrf': {
            'lambda_param': 0.5,  # 平滑项强度 (建议0.3-0.8)
            'connectivity': 6     # 邻域连接方式
        },
        'morphology': {
            'min_size': 8,        # 最小保留体积 (体素数)
            'close_r': 2          # 闭运算结构球半径
        },
        'threshold': 0.4          # 最终概率阈值
    }

    # 加载模型输出的概率图 (示例)
    # prob_volume = np.load("vascular_prob.npy")  
    
    # Step 1: MRF优化
    binary_volume = mrf_probability_refinement(prob_volume, **pipeline['mrf'])
    
    # Step 2: 形态学处理
    refined_volume = morphology_postprocess(binary_volume, **pipeline['morphology'])
    
    # Step 3: 最终阈值处理
    final_result = (prob_volume > pipeline['threshold']).astype(np.uint8)
    final_result[refined_volume == 1] = 1  # 保留形态学优化结果
    
    return final_result
