import pytest

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.dataset import Dataset
from torch_frame.datasets import FakeDataset
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearBucketEncoder,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.testing.text_embedder import HashTextEmbedder


@pytest.mark.parametrize('encoder_cat_cls_kwargs', [(EmbeddingEncoder, {})])
@pytest.mark.parametrize('encoder_num_cls_kwargs', [
    (LinearEncoder, {}),
    (LinearBucketEncoder, {}),
    (LinearPeriodicEncoder, {
        'n_bins': 4
    }),
])
@pytest.mark.parametrize('encoder_text_embedded_cls_kwargs', [
    (LinearEmbeddingEncoder, {
        'in_channels': 12
    }),
])
def test_stypewise_feature_encoder(
    encoder_cat_cls_kwargs,
    encoder_num_cls_kwargs,
    encoder_text_embedded_cls_kwargs,
):
    num_rows = 10
    dataset: Dataset = FakeDataset(
        num_rows=num_rows, with_nan=False,
        stypes=[stype.categorical, stype.numerical,
                stype.text_embedded], text_embedder_cfg=TextEmbedderConfig(
                    text_embedder=HashTextEmbedder(
                        encoder_text_embedded_cls_kwargs[1]['in_channels']),
                    batch_size=None))
    dataset.materialize()
    tensor_frame = dataset.tensor_frame
    out_channels = 8

    encoder = StypeWiseFeatureEncoder(
        out_channels=out_channels,
        col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
        stype_encoder_dict={
            stype.categorical:
            encoder_cat_cls_kwargs[0](**encoder_cat_cls_kwargs[1]),
            stype.numerical:
            encoder_num_cls_kwargs[0](**encoder_num_cls_kwargs[1]),
            stype.text_embedded:
            encoder_text_embedded_cls_kwargs[0](
                **encoder_text_embedded_cls_kwargs[1]),
        },
    )
    x, col_names = encoder(tensor_frame)
    assert x.shape == (num_rows, tensor_frame.num_cols, out_channels)
    assert col_names == ['a', 'b', 'c', 'x', 'y', 'text_1', 'text_2']
