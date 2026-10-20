"""Persistent experiment history and configuration-based result cache."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4


class RunLogStore:
    """Store complete attack responses as JSON files under the project root."""

    # Increment when the evaluation algorithm changes so stale metric payloads
    # remain viewable in history but are never returned as a current cache hit.
    CACHE_VERSION = 3

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.records_dir = self.root_dir / "records"
        self.index_path = self.root_dir / "index.json"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache_key(attack_id: str, data_path: str, params: dict) -> str:
        payload = {
            "cache_version": RunLogStore.CACHE_VERSION,
            "attack_id": attack_id,
            "data_path": str(Path(data_path).expanduser().resolve()),
            "params": params,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(canonical.encode("utf-8")).hexdigest()

    def find_by_cache_key(self, cache_key: str) -> dict | None:
        for summary in reversed(self._read_index().get("runs", [])):
            if summary.get("cache_key") == cache_key:
                return self.get(summary.get("id"))
        return None

    def get(self, run_id: str | None) -> dict | None:
        if not run_id:
            return None
        path = self.records_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return self._read_json(path, None)

    def list(self) -> list[dict]:
        return list(reversed(self._read_index().get("runs", [])))

    def touch(self, record: dict) -> dict:
        record["last_accessed_at"] = self._timestamp()
        record["cache_hit_count"] = int(record.get("cache_hit_count", 0)) + 1
        self._write_json(self.records_dir / f"{record['id']}.json", record)

        index = self._read_index()
        for summary in index.get("runs", []):
            if summary.get("id") == record["id"]:
                summary.update(self._summary(record))
                break
        self._write_index(index)
        return record

    def save(self, *, cache_key: str, attack_id: str, data_path: str, params: dict, result: dict) -> dict:
        timestamp = self._timestamp()
        record = {
            "id": uuid4().hex,
            "cache_key": cache_key,
            "attack_id": attack_id,
            "data_path": data_path,
            "params": params,
            "result": result,
            "created_at": timestamp,
            "last_accessed_at": timestamp,
            "cache_hit_count": 0,
        }
        self._write_json(self.records_dir / f"{record['id']}.json", record)

        index = self._read_index()
        index.setdefault("runs", []).append(self._summary(record))
        self._write_index(index)
        return record

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _summary(record: dict) -> dict:
        result = record.get("result", {})
        return {
            "id": record["id"],
            "cache_key": record["cache_key"],
            "attack_id": record["attack_id"],
            "data_path": record["data_path"],
            "params": record.get("params", {}),
            "metrics": result.get("metrics", {}),
            "created_at": record["created_at"],
            "last_accessed_at": record.get("last_accessed_at", record["created_at"]),
            "cache_hit_count": record.get("cache_hit_count", 0),
        }

    def _read_index(self) -> dict:
        return self._read_json(self.index_path, {"version": 1, "runs": []})

    def _write_index(self, index: dict) -> None:
        self._write_json(self.index_path, index)

    @staticmethod
    def _read_json(path: Path, fallback):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        temp_path.replace(path)
