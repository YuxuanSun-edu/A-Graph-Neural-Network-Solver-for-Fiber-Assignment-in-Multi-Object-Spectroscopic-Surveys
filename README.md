# A-Graph-Neural-Network-Approach-for-Fast-Fiber-Assignment-in-Multi-Object-Spectroscopic-Surveys
这里是论文"A Graph Neural Network Approach for Fast Fiber Assignment in Multi-Object Spectroscopic Surveys"的代码库
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

如果你想循环自动化训练（使用我们的自动化数据生成）
--python auto_train_loop.py

如果你想进行推理过程
python pipeline.py --dir [你的数据文件地址] --model [你的模型文件地址]

如果你想使用自己的，你的数据文件应该包含以下两种文件
--{id} items.txt - 候选观测任务
格式形如
```text
#n sky+std+addon None obj 15000 weight 4.50000000E+07
F1 G1001: 3500.50
F1 G1002: 4200.75
F2 G1003: 2800.25
...

{id} limits.txt - 冲突约束组
格式形如
```text
#n sky+std+addon None obj None weight None
F1 G1001, F1 G1002          # 同一光纤不能同时观测两颗星
F1 G1001, F2 G1001          # 同一颗星不能被两个光纤同时观测
F3 G2001, F4 G2002          # 机械碰撞约束（距离过近）
