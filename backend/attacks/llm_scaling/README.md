┌──────────────────────────────────────────────────────────────────────────────┐
│                        原始文件 → 新框架文件 映射表                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  precompute_collection_mia_scores.py                                         │
│  ├── calculatePerplexity()     → utils.py :: calculate_perplexity()          │
│  ├── inference()               → utils.py :: compute_mia_features()          │
│  └── __main__(加载模型/数据/   → precompute_collection.py :: main()           │
│       循环计算/保存jsonl)                                                     │
│                                                                              │
│  precompute_document_mia_scores.py                                           │
│  ├── calculatePerplexity()     → utils.py :: calculate_perplexity() [复用]   │
│  ├── inference()               → utils.py :: compute_mia_features() [复用]   │
│  ├── create_chunks()           → precompute_document.py :: create_chunks()   │
│  └── __main__(加载模型/数据/   → precompute_document.py :: main()             │
│       切chunk/计算/保存)                                                      │
│                                                                              │
│  collection_mia.py                                                           │
│  ├── combinations()            → utils.py :: combinations_from_list()         │
│  ├── process_combination_      → utils.py :: process_combination_features()   │
│  │   mia_features()                                                          │
│  ├── train_model()             → utils.py :: train_linear_scorer()            │
│  ├── evaluate_model()          → utils.py :: evaluate_chunk_auc()             │
│  ├── get_raw_predictions()     → utils.py :: predict_raw()                    │
│  ├── get_tpr_fpr()             → metrics.py :: get_tpr_fpr()                  │
│  ├── compute_auc()             → metrics.py :: compute_best_metrics()         │
│  ├── run_dataset_inference_    → attack_collection.py :: run()                │
│  │   pvalues() [核心逻辑]                                                    │
│  └── __main__(多参数循环)       → 由前端单次API调用替代                         │
│                                                                              │
│  document_mia.py                                                             │
│  ├── combinations()            → utils.py :: combinations_from_list() [复用]  │
│  ├── process_combination_      → utils.py :: process_combination_features()   │
│  │   mia_features()              [复用]                                      │
│  ├── train_model()             → utils.py :: train_linear_scorer() [复用]     │
│  ├── evaluate_model()          → utils.py :: evaluate_chunk_auc() [复用]      │
│  ├── get_raw_predictions()     → utils.py :: predict_raw() [复用]             │
│  ├── get_tpr_fpr()             → metrics.py :: get_tpr_fpr() [复用]           │
│  ├── get_mia_scorer()          → attack_document.py :: run()内联              │
│  ├── dict_to_tensor()          → 在run()中直接用numpy处理                     │
│  ├── convert_docs2tensors()    → 在run()中直接用numpy处理                     │
│  ├── get_eval_scores()         → attack_document.py :: run()内部              │
│  ├── evaluate()                → attack_document.py :: run()内联              │
│  │   [Mann-Whitney U test]       + metrics.py :: compute_best_metrics()      │
│  ├── clean_nan()               → utils.py :: clean_nan_nested()               │
│  ├── run_mia() [核心逻辑]      → attack_document.py :: run()                  │
│  └── __main__(多模型/数据集     → 由前端单次API调用替代                         │
│       /chunk_size/seed循环)                                                   │
│                                                                              │
│  [新增] 原始无对应                                                            │
│  ├── base.py                   ← 新增：抽象基类 + AttackResult                │
│  ├── registry.py               ← 新增：攻击注册中心                           │
│  ├── visualizer.py             ← 新增：图表数据生成                           │
│  ├── routes.py                 ← 新增：REST API路由                           │
│  └── app.py                    ← 新增：Flask入口                              │
└──────────────────────────────────────────────────────────────────────────────┘