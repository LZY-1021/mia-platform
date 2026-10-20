def register_attacks():
    """
    Register LLM scaling attacks for the Flask app.

    Keep imports lazy so CLI precompute modules can run without importing the
    dashboard-only attack classes and their core.visualizer/core.metrics deps.
    """
    from attacks.llm_scaling.attack_collection import LLMScalingCollectionAttack
    from attacks.llm_scaling.attack_document import LLMScalingDocumentAttack
    from attacks.llm_scaling.attack_whitebox import LLMScalingWhiteboxAttack
    from attacks.registry import register

    register(LLMScalingCollectionAttack())
    register(LLMScalingDocumentAttack())
    register(LLMScalingWhiteboxAttack())
