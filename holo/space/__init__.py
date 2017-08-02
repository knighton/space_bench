from .annoy import AnnoySpace
from .brute import BruteSpace
from .faiss import BruteFaissSpace, LSHFaissSpace
from .nmslib import NMSLibSpace


NAME2CLASS = {
    AnnoySpace.name: AnnoySpace,
    BruteFaissSpace.name: BruteFaissSpace,
    BruteSpace.name: BruteSpace,
    LSHFaissSpace.name: LSHFaissSpace,
    NMSLibSpace.name: NMSLibSpace,
}


def get(name):
    return NAME2CLASS[name]
