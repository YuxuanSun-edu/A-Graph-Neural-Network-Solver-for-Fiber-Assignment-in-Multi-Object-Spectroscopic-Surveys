# A-Graph-Neural-Network-Approach-for-Fast-Fiber-Assignment-in-Multi-Object-Spectroscopic-Surveys
Code for A Graph Neural Network Approach for Fast Fiber Assignment in Multi-Object Spectroscopic Surveys

```text
├── data_generator.py      # 基于 LAMOST 物理参数生成随机观测场景
├── build_gnn_sample.py    # 图缩减 (Reduction) 与 6 维特征工程
├── model.py               # GNN 模型，基于 PyG 的 MISScoreGNN 网络结构
├── gnn_sampler_gpu.py     # GPU 并行采样、冲突过滤与 ILS 局部搜索
├── auto_train_loop.py     # 自动化数据生成 → 标注 → 训练 → 备份循环
├── pipeline.py            # 集成化一键推理与报表生成工具
├── train.py               # 模型训练与微调管线
├── main_gnn_solver.py     # 单图求解入口 
├── reduction.py           # 权重支配规则图缩减与冲突图构建
├── CHILS.py               # 并发混合迭代局部搜索 (教师模型)
├── TSO.py                 # 基于网络流的基线算法 (Two-Stage Optimization)
├── compare.py             # 多算法对比测试框架 (Greedy/SA/GA)
└── build.py               # 批量构建训练数据的命令行工具
