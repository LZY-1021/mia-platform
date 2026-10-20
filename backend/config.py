import os
import torch


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "mia-demo-secret")
    # results 目录在项目根目录（backend 的上一级）
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    RUN_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_logs")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG = True
