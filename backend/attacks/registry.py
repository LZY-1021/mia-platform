"""攻击注册中心。新增模块。"""

from typing import Dict, List
from attacks.base import BaseAttack

_REGISTRY: Dict[str, BaseAttack] = {}


def register(attack: BaseAttack):
    _REGISTRY[attack.name] = attack
    return attack


def get_attack(name: str) -> BaseAttack:
    if name not in _REGISTRY:
        raise KeyError(f"Attack '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_attacks() -> List[dict]:
    return [atk.info() for atk in _REGISTRY.values()]