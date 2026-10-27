"""Compatibility import; all dataset validation is shared in person_data.py."""

from .person_data import PersonDataset


class RSTPReid(PersonDataset):
    def __init__(self, root="", verbose=True, **kwargs):
        super().__init__(root, "RSTPReid", **kwargs)


RSTPReid_ThreeModal = RSTPReid
