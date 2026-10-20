import math
import traceback
from flask import Blueprint, current_app, jsonify, request
from attacks.registry import get_attack, list_attacks
from core.run_log_store import RunLogStore

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _json_safe(value):
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except ValueError:
            pass
    return value


def _run_log_store():
    return RunLogStore(current_app.config["RUN_LOG_DIR"])


@api_bp.route("/attacks", methods=["GET"])
def get_attacks():
    """列出所有已注册的攻击"""
    attacks = list_attacks()
    print("Registered attacks:", attacks)
    return jsonify(attacks)
    #return jsonify(list_attacks())


@api_bp.route("/attacks/<attack_id>/params", methods=["GET"])
def get_params(attack_id):
    try:
        atk = get_attack(attack_id)
        return jsonify(atk.get_default_params())
    except KeyError as e:
        return jsonify({"error": str(e)}), 404


@api_bp.route("/attacks/<attack_id>/data", methods=["GET"])
def get_data(attack_id):
    """直接调用攻击类的 discover_data 方法，返回其结果"""
    try:
        atk = get_attack(attack_id)
        results_dir = current_app.config.get("RESULTS_DIR", "results")
        data_structure = atk.discover_data(results_dir)
        return jsonify(data_structure)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_bp.route("/attacks/<attack_id>/run", methods=["POST"])
def run_attack(attack_id):
    try:
        atk = get_attack(attack_id)
        body = request.get_json(force=True)
        data_path = body.get("data_path")
        params = body.get("params", atk.get_default_params())
        # 是否调用实验日志缓存：默认为 True（命中则直接返回历史结果，否则重新运行）
        use_cache = bool(body.get("use_cache", True))

        print(attack_id)
        print(data_path)

        if not data_path:
            return jsonify({"error": "data_path is required"}), 400

        store = _run_log_store()
        cache_key = store.cache_key(attack_id, data_path, params)
        if use_cache:
            cached_record = store.find_by_cache_key(cache_key)
            if cached_record:
                cached_record = store.touch(cached_record)
                cached_result = cached_record["result"].copy()
                cached_result.update({
                    "run_id": cached_record["id"],
                    "cached": True,
                    "cache_hit_count": cached_record["cache_hit_count"],
                })
                return jsonify(_json_safe(cached_result))
            
        if attack_id=='vlm_reference_inference_attack':
            result=atk.run(
                data_path,
                target_similarity_file=params["target_similarity_file"],
                non_member_reference_similarity_file=params["non_member_reference_similarity_file"],
                temperature=params["temperature"],
                metric=params["similarity_metric"],
                granularity=params["granularity"],
                n_rounds=params["n_rounds"],
            );
            print(result)
            
        else :
            scores = atk.load_scores(data_path)
            result = atk.run(scores, params)

        result_payload = _json_safe({
            "attack_name": result.attack_name,
            "metrics": result.metrics,
            "charts": result.charts,
            "params": result.params,
            "extra": result.extra,
            "num_scores": len(result.scores),
        })
        record = store.save(
            cache_key=cache_key,
            attack_id=attack_id,
            data_path=data_path,
            params=_json_safe(params),
            result=result_payload,
        )
        result_payload.update({"run_id": record["id"], "cached": False, "cache_hit_count": 0})
        return jsonify(result_payload)
    except NotImplementedError as e:
        return jsonify({"error": f"Attack not implemented: {e}"}), 501
    except FileNotFoundError as e:
        return jsonify({"error": f"Data not found: {e}"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_bp.route("/runs", methods=["GET"])
def get_run_logs():
    """Return persistent experiment summaries, newest first."""
    return jsonify({"runs": _run_log_store().list()})


@api_bp.route("/runs/<run_id>", methods=["GET"])
def get_run_log(run_id):
    """Return one full experiment record, including its rendered chart payload."""
    record = _run_log_store().get(run_id)
    if not record:
        return jsonify({"error": "Experiment log not found"}), 404
    return jsonify(_json_safe(record))
