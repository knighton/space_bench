from .annoy import AnnoySpace
from .faiss import FaissSpace
from .nmslib import NMSLibSpace


NAME2CLASS = {
    AnnoySpace.name: AnnoySpace,
    FaissSpace.name: FaissSpace,
    NMSLibSpace.name: NMSLibSpace,
}


def get(name):
    return NAME2CLASS[name]
