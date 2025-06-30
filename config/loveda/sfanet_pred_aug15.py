from network.models.SFANet import SFANet
from network.losses import UnetFormerLoss
from network.datasets.loveda_dataset import CLASSES


ignore_index = len(CLASSES)
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "sfanet_v4(aug_15)"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "sfanet"
log_name = 'loveda/{}'.format(weights_name)

test_dataset = None

net = SFANet(num_classes=num_classes)
loss = UnetFormerLoss(ignore_index=ignore_index)
