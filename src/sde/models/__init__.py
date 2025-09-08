from ..models.classical.logreg import LOGISTIC_REGRESSION_MODEL
from ..models.classical.mlp import MLP_MODEL
from ..models.vision.cnn import SIMPLE_CNN_MODEL
from ..models.vision.resnet import CONFIGURABLE_RESNET_MODEL

AVAILABLE_MODELS = {
    LOGISTIC_REGRESSION_MODEL.name: LOGISTIC_REGRESSION_MODEL,
    MLP_MODEL.name: MLP_MODEL,
    SIMPLE_CNN_MODEL.name: SIMPLE_CNN_MODEL,
    CONFIGURABLE_RESNET_MODEL.name: CONFIGURABLE_RESNET_MODEL,
}
