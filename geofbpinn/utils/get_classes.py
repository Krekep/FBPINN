from geofbpinn.geometry import DecompositionND
from geofbpinn.geometry.base_decomposition import BaseDecomposition
from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon
from geofbpinn.networks.embeddings import BaseEmbedding, FourierEmbedding
from geofbpinn.networks.schedulers.layer import (
    BaseLayerScheduler,
    SequenceBlockScheduler,
    ReduceEpochsBlockScheduler,
    SequenceLayerScheduler,
    GrowingColumnScheduler,
    TwoStepLayerScheduler,
)
from geofbpinn.networks.schedulers.loss import LossScheduler, AdaptiveLossScheduler
from geofbpinn.networks.schedulers.lr import WarmupReduceLROnPlateau


def get_decomposition(name: str) -> type(BaseDecomposition):
    classes = {
        "Decomposition2DPolygon": Decomposition2DPolygon,
        "DecompositionND": DecompositionND,
    }
    return classes[name]


def get_embedding(name: str) -> type(BaseEmbedding):
    classes = {"Base": BaseEmbedding, "Fourier": FourierEmbedding}
    return classes[name]


def get_layer_scheduler(name: str) -> type(BaseLayerScheduler):
    classes = {
        "Base": BaseLayerScheduler,
        "SequenceBlock": SequenceBlockScheduler,
        "ReduceEpochsBlock": ReduceEpochsBlockScheduler,
        "SequenceLayer": SequenceLayerScheduler,
        "GrowingColumn": GrowingColumnScheduler,
        "TwoStepLayer": TwoStepLayerScheduler,
    }
    return classes[name]


def get_loss_scheduler(name: str) -> type(LossScheduler):
    classes = {
        "Base": LossScheduler,
        "Adaptive": AdaptiveLossScheduler,
    }
    return classes[name]


def get_lr_scheduler(name: str):
    classes = {
        "WarmupReduceLROnPlateau": WarmupReduceLROnPlateau,
    }
    return classes[name]
