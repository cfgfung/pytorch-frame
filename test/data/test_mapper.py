import pandas as pd
import torch

from torch_frame.data.mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    TextEmbeddingTensorMapper,
    TimeTensorMapper,
)
from torch_frame.testing.text_embedder import HashTextEmbedder


def test_numerical_tensor_mapper():
    ser = pd.Series([0.0, 10.0, float('NaN'), 30.0])
    expected = torch.tensor([0.0, 10.0, float('NaN'), 30.0])

    mapper = NumericalTensorMapper()

    out = mapper.forward(ser)
    assert out.dtype == torch.float
    assert torch.equal(out.isnan(), expected.isnan())
    assert torch.equal(out.nan_to_num(), expected.nan_to_num())

    out = mapper.backward(out)
    pd.testing.assert_series_equal(out, ser, check_dtype=False)


def test_categorical_tensor_mapper():
    ser = pd.Series(['A', 'B', None, 'C', 'B'])
    expected = torch.tensor([1, 0, -1, -1, 0])

    mapper = CategoricalTensorMapper(['B', 'A'])

    out = mapper.forward(ser)
    assert out.dtype == torch.long
    assert torch.equal(out, expected)

    out = mapper.backward(out)
    pd.testing.assert_series_equal(out, pd.Series(['A', 'B', None, None, 'B']))


def test_multicategorical_tensor_mapper():
    ser = pd.Series(['A,B', 'B', '', 'C', 'B,C', None])
    expected_values = torch.tensor([1, 0, 0, 0, -1])
    expected_boundaries = torch.tensor([0, 2, 3, 3, 3, 4, 5])
    mapper = MultiCategoricalTensorMapper(['B', 'A'], sep=",")

    tensor = mapper.forward(ser)
    values = tensor.values
    offset = tensor.offset
    assert values.dtype == torch.long
    assert torch.equal(
        values[expected_boundaries[0]:expected_boundaries[1]].sort().values,
        torch.tensor([0, 1]))
    assert torch.equal(values[expected_boundaries[1]:],
                       expected_values[expected_boundaries[1]:])
    assert torch.equal(offset, expected_boundaries)

    out = mapper.backward(tensor)
    assert out.values[0] == 'A,B' or out.values[0] == 'B,A'
    assert out.values[1] == 'B'
    assert out.values[2] == ''
    assert out.values[3] == ''
    assert out.values[4] == 'B'
    assert out.values[5] == ''


def test_text_embedding_tensor_mapper():
    out_channels = 10
    num_sentences = 20
    ser = pd.Series(["Hello world!"] * (num_sentences // 2) +
                    ["I love torch-frame"] * (num_sentences // 2) + [0.1])
    mapper = TextEmbeddingTensorMapper(HashTextEmbedder(out_channels),
                                       batch_size=8)
    emb = mapper.forward(ser)
    assert emb.shape == (num_sentences + 1, out_channels)
    mapper.batch_size = None
    emb2 = mapper.forward(ser)
    assert torch.allclose(emb, emb2)


def test_time_tensor_mapper():
    expected_values = torch.tensor([1597276800000000000, 1697760000000000000, 
                                    1698624000000000000, -9223372036854775808], dtype=torch.int64)
    timestr = ['2020-08-13', '2023-10-20', '2023-10-30', 'nan']
    ser = pd.Series(timestr)
    mapper = TimeTensorMapper("%Y-%m-%d")
    out = mapper.forward(ser)
    assert out.dtype == torch.int64
    assert torch.equal(out, expected_values)
    recovered = mapper.backward(out)
    assert recovered.equals(ser)

    timestr2 = ['08-13-2020', '10-20-2023', '10-30-2023', 'nan']
    ser2 = pd.Series(timestr2)
    mapper2 = TimeTensorMapper("%m-%d-%Y")
    out2 = mapper2.forward(ser2)
    assert out2.dtype == torch.int64
    assert torch.equal(out2, expected_values)
    recovered2 = mapper2.backward(out2)
    assert recovered.equals(ser2)