"""
所有 MIA 攻击的抽象基类。新增模块。

统一数据流:
  Raw Data → Precompute Scores → Train Scorer → Aggregate & Predict → Evaluate & Visualize
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AttackResult:
    attack_name: str
    metrics: Dict[str, float]
    scores: List[float]
    labels: List[int]
    params: Dict[str, Any]
    charts: List[Dict] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    name: str = "base"
    display_name: str = "Base Attack"
    description: str = ""
    category: str = ""
    access: str = ""

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        """
        返回数据发现结果。
        
        可以返回两种格式：
        1. 简单列表格式（兼容旧接口）:
           {
             "type": "simple",
             "data": [{"label": "...", "path": "..."}, ...]
           }
        
        2. 层级结构格式（新接口）:
           {
             "type": "hierarchical",
             "levels": ["model", "dataset", "chunk_size"],  # 层级名称
             "data": { ... }  # 层级数据
           }
        """
        ...

    @abstractmethod
    def load_scores(self, data_path: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        ...

    def info(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "access": self.access,
        }
    """所有 MIA 攻击的抽象基类。"""
    name: str = "base"
    display_name: str = "Base Attack"
    description: str = ""
    category: str = ""       # "llm" | "vlm"
    access: str = ""         # "blackbox" | "whitebox"

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def discover_data(self, results_dir: str) -> List[Dict[str, str]]:
        ...

    @abstractmethod
    def load_scores(self, data_path: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        ...

    def info(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "access": self.access,
        }