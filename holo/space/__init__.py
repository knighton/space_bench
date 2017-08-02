from .annoy import AnnoySpace
from .brute import BruteSpace
from .faiss import FaissSpace
from .nmslib import NMSLibSpace


NAME2CLASS = {
    AnnoySpace.name: AnnoySpace,
    BruteSpace.name: BruteSpace,
    FaissSpace.name: FaissSpace,
    NMSLibSpace.name: NMSLibSpace,
}


def get(name):
    return NAME2CLASS[name]
