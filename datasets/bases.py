from .person_data import (
    PersonDataset as BaseDataset,
    TrainDataset,
    GalleryDataset,
    QueryDataset,
    tokenize,
)

ImageTextDataset = TrainDataset
ImageTextSketchDataset = TrainDataset
