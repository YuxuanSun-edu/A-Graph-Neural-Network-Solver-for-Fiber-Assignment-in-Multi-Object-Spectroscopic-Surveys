import os
import random
import copy
from concurrent.futures import ProcessPoolExecutor
from typing import List, Set, Tuple

from reduction import ProblemInstance, Reducer, load_items_from_file, load_limits_from_file

# ============================================================
# 1. 核心算子：更强的局部搜索 (1-for-k Swap)
# ============================================================

def local_search_refinement(instance: ProblemInstance, current_sol: Set[int], adj: List[Set[int]]) -> Set[int]:
    """
    改进当前的独立集：通过 1-for-k 交换寻找更高权重的邻域解
    """
    items = instance.items
    weights = [it[2] for it in items]
    n = len(items)
    
    improved = True
    best_sol = set(current_sol)
    best_w = sum(weights[i] for i in best_sol)

    while improved:
        improved = False
        # 遍历所有不在当前解中的点
        candidates = list(set(range(n)) - best_sol)
        # 启发式排序：优先尝试高权重的点
        candidates.sort(key=lambda u: weights[u], reverse=True)

        for u in candidates:
            # 找到与 u 冲突的当前解中的节点
            conflicts = [v for v in adj[u] if v in best_sol]
            conflicts_w = sum(weights[v] for v in conflicts)

            # 如果加入 u 并移除冲突点后，总权重增加，则执行交换
            if weights[u] > conflicts_w:
                for v in conflicts:
                    best_sol.remove(v)
                best_sol.add(u)
                best_w = best_w - conflicts_w + weights[u]
                improved = True
                break # 立即进入下一轮迭代
    return best_sol

# ============================================================
# 2. 扰动算子 (Perturbation)：跳出局部最优的关键
# ============================================================

def perturb(current_sol: Set[int], adj: List[Set[int]], strength: float = 0.1) -> Set[int]:
    """
    随机移除一部分节点，并重新通过贪心填补，产生新的搜索起点
    """
    new_sol = set(current_sol)
    num_to_remove = max(1, int(len(current_sol) * strength))
    
    # 随机移除
    to_remove = random.sample(list(current_sol), num_to_remove)
    for r in to_remove:
        new_sol.remove(r)
        
    return new_sol

def build_adj_list(instance):
    """
    根据 groups 构造图的邻接表：
      - 节点：0..n-1
      - 每个 group 视作 clique，在组内两两连边
    """
    n = len(instance.items)
    groups = instance.groups

    adj = [set() for _ in range(n)]
    for g in groups:
        for i in range(len(g)):
            u = g[i]
            for j in range(i + 1, len(g)):
                v = g[j]
                if u == v:
                    continue
                adj[u].add(v)
                adj[v].add(u)
    return adj

# ============================================================
# 3. 单线程 ILS 实例 (Worker)
# ============================================================

def ils_worker(instance_data: Tuple[ProblemInstance, List[Set[int]], int, int]) -> Set[int]:
    """
    单个进程执行的迭代局部搜索
    """
    instance, adj, max_iters, seed = instance_data
    random.seed(seed)
    
    # 1. 初始解生成 (使用带随机扰动的加权贪心)
    items = instance.items
    weights = [it[2] for it in items]
    n = len(items)
    
    def get_initial():
        sol = set()
        rem = set(range(n))
        while rem:
            u = max(rem, key=lambda x: weights[x] / (1.0 + len(adj[x]) + random.random()*0.1))
            sol.add(u)
            rem -= ({u} | adj[u])
        return sol

    current_sol = get_initial()
    current_sol = local_search_refinement(instance, current_sol, adj)
    best_sol = set(current_sol)
    best_w = sum(weights[i] for i in best_sol)

    # 2. 迭代环
    for _ in range(max_iters):
        # 扰动 + 局部搜索
        perturbed_sol = perturb(current_sol, adj)
        # 填补由于扰动留下的空间
        rem = set(range(n)) - perturbed_sol
        for node in perturbed_sol:
            rem -= adj[node]
        while rem:
            u = max(rem, key=lambda x: weights[x] / (1.0 + len(adj[x])))
            perturbed_sol.add(u)
            rem -= ({u} | adj[u])
            
        candidate_sol = local_search_refinement(instance, perturbed_sol, adj)
        candidate_w = sum(weights[i] for i in candidate_sol)

        # 接受准则 (这里使用单纯的爬山，也可以改用类似 SA 的概率接受)
        if candidate_w > best_w:
            best_sol = set(candidate_sol)
            best_w = candidate_w
            current_sol = set(candidate_sol)
        
    return best_sol

# ============================================================
# 4. 真正的并发 CHILS 入口
# ============================================================

def chils_solver(instance: ProblemInstance, num_workers: int = 4, iters_per_worker: int = 100):
    """
    Concurrent Hybrid Iterated Local Search
    """
    adj = build_adj_list(instance)
    
    print(f"[CHILS] 启动 {num_workers} 个并发搜索进程...")
    
    # 准备每个进程的参数
    worker_args = [(instance, adj, iters_per_worker, random.randint(0, 10000)) for _ in range(num_workers)]
    
    best_global_sol = set()
    best_global_w = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(ils_worker, worker_args))
        
    weights = [it[2] for it in instance.items]
    for sol in results:
        w = sum(weights[i] for i in sol)
        if w > best_global_w:
            best_global_w = w
            best_global_sol = sol
            
    return sorted(list(best_global_sol))
