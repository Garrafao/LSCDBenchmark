import numpy as np

from src.config import pairing, sampling
from src.vector_model import VectorModel
from src.lscd.target import Target


def apd_compare_all(target: Target, model: VectorModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.all, pairing=pairing.COMPARE)).item()


def apd_all_all(target: Target, model: VectorModel) -> float:
    distances = model.distances(target=target, sampling=sampling.all, pairing=pairing.COMPARE) \
                + model.distances(target=target, sampling=sampling.all, pairing=pairing.LATER) \
                + model.distances(target=target, sampling=sampling.all, pairing=pairing.EARLIER)
    return np.mean(distances).item()


def apd_earlier_all(target: Target, model: VectorModel) -> float:
    return np.mean(model.distances(sampling.all(pairing.EARLIER, target)))


def apd_later_all(target: Target, model: VectorModel) -> float:
    return np.mean(model.distances(sampling.all(pairing.LATER, target)))


def apd_compare_annotated(target: Target, model: VectorModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.annotated, pairing=pairing.COMPARE)).item()

def apd_all_annotated(target: Target, model: VectorModel) -> float:
    distances = model.distances(target=target, sampling=sampling.annotated, pairing=pairing.COMPARE) \
                + model.distances(target=target, sampling=sampling.annotated, pairing=pairing.LATER) \
                + model.distances(target=target, sampling=sampling.annotated, pairing=pairing.EARLIER)
    return np.mean(distances).item()

def apd_later_annotated(target: Target, model: VectorModel) -> float:
    return np.mean(model.distances(sampling.annotated(pairing.LATER, target)))


def apd_earlier_annotated(target: Target, model: VectorModel) -> float:
    return np.mean(model.distances(sampling.annotated(pairing.EARLIER, target)))


def apd_compare_sampled(
        target: Target, model: VectorModel, n: int, replace: bool
) -> float:
    return np.mean(
        model.distances(sampling.sampled(pairing.COMPARE, target, n=n, replace=replace))
    )


def apd_compare_sampled(
        target: Target, model: VectorModel, n: int, replace: bool
) -> float:
    return np.mean(
        model.distances(sampling.sampled(pairing.COMPARE, target, n=n, replace=replace))
    )


def apd_compare_all_minus_all_annotated(target: Target, model: VectorModel) -> float:
    return np.mean(
        model.distances(sampling.annotated(pairing.COMPARE, target))
    ) - np.mean(
        model.distances(
            sampling.annotated(pairing.COMPARE, target)
            + sampling.annotated(pairing.LATER, target)
            + sampling.annotated(pairing.EARLIER, target)
        )
    )


diasense = apd_compare_all_minus_all_annotated
