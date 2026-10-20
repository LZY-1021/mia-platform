from attacks.resnet_lira_whitebox.attack import ResNetLiRAWhiteBoxAttack
from attacks.registry import register

register(ResNetLiRAWhiteBoxAttack())
