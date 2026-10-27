"""Compatibility import; all dataset validation is shared in person_data.py."""

from .person_data import PersonDataset


class ICFG_PEDES(PersonDataset):
    def __init__(self, root="", verbose=True, **kwargs):
        super().__init__(root, "ICFG-PEDES", **kwargs)


ICFG_PEDES_ThreeModal = ICFG_PEDES
