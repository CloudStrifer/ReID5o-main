"""Compatibility import; all dataset validation is shared in person_data.py."""

from .person_data import PersonDataset


class CUHK_PEDES(PersonDataset):
    def __init__(self, root="", verbose=True, **kwargs):
        super().__init__(root, "CUHK-PEDES", **kwargs)


CUHK_PEDES_ThreeModal = CUHK_PEDES
