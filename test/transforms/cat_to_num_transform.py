import pytest
import torch
import torch.nn.functional as F

from torch_frame import TaskType, TensorFrame, stype
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_frame.datasets.fake import FakeDataset
from torch_frame.transforms import CatToNumTransform


@pytest.mark.parametrize('with_nan', [True, False])
def test_ordered_target_statistics_encoder_on_categorical_only_dataset(
        with_nan):
    num_rows = 10
    dataset: Dataset = FakeDataset(
        num_rows=num_rows, with_nan=with_nan, stypes=[stype.categorical],
        task_type=TaskType.MULTICLASS_CLASSIFICATION, create_split=True)
    dataset.df['x'] = 0
    dataset.materialize()
    total_cols = len(dataset.feat_cols)
    categorical_features = dataset.tensor_frame.col_names_dict[
        stype.categorical]
    # the dataset only contains categorical features
    assert (total_cols == len(categorical_features))
    target = F.one_hot(dataset.tensor_frame.y, 3)
    target_mean = target.float().mean(dim=0)
    tensor_frame: TensorFrame = dataset.tensor_frame
    for col in dataset.feat_cols:
        col_stats = dataset.col_stats[col]
        # all the original columns are categorical
        for stat in StatType.stats_for_stype(stype.numerical):
            assert (stat not in col_stats)
        for stat in StatType.stats_for_stype(stype.categorical):
            assert (stat in col_stats)
    transform = CatToNumTransform()
    transform.fit(dataset.tensor_frame, dataset.col_stats)
    transformed_col_stats = transform.transformed_stats
    for col in transformed_col_stats:
        # ensure that the transformed col stats contain
        # only numerical col stats
        for stat in StatType.stats_for_stype(stype.numerical):
            assert (stat in transformed_col_stats[col])
        for stat in StatType.stats_for_stype(stype.categorical):
            assert (stat not in transformed_col_stats[col])
    out = transform(tensor_frame)

    # The first categorical column, x, is changed to all 0's initially.
    # This assert statement makes sure that transform is correct.
    # The transform uses m-target estimation, where each categorical
    # feature is transformed to (num_count + target_mean)/(num_rows + 1).
    # This test tests the correctness in multiclass classification task.
    assert torch.allclose(
        out.feat_dict[stype.numerical][:, 0].float(),
        torch.tensor((num_rows + target_mean[0]) / (num_rows + 1),
                     device=out.device).repeat(num_rows))

    # assert that there are no categorical features
    assert (stype.categorical not in out.col_names_dict)
    assert (stype.categorical not in out.feat_dict)

    assert (len(
        out.col_names_dict[stype.numerical]) == ((dataset.num_classes - 1) *
                                                 total_cols))


@pytest.mark.parametrize('task_type', [
    TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION,
    TaskType.BINARY_CLASSIFICATION
])
def test_ordered_target_statistics_encoder(task_type):
    num_rows = 10
    dataset: Dataset = FakeDataset(num_rows=num_rows, with_nan=False,
                                   stypes=[stype.numerical, stype.categorical],
                                   task_type=task_type, create_split=True)
    dataset.df['x'] = 0
    dataset.materialize()
    total_cols = len(dataset.feat_cols)
    total_numerical_cols = len(
        dataset.tensor_frame.col_names_dict[stype.numerical])
    tensor_frame: TensorFrame = dataset.tensor_frame
    transform = CatToNumTransform()
    transform.fit(dataset.tensor_frame, dataset.col_stats)
    transformed_col_stats = transform.transformed_stats
    for col in transformed_col_stats:
        # ensure that the transformed col stats contain
        # only numerical col stats
        for stat in StatType.stats_for_stype(stype.numerical):
            assert (stat in transformed_col_stats[col])
        for stat in StatType.stats_for_stype(stype.categorical):
            assert (stat not in transformed_col_stats[col])
    out = transform(tensor_frame)
    # assert that there are no categorical features
    assert (stype.categorical not in out.col_names_dict)
    assert (stype.categorical not in out.feat_dict)

    if task_type != TaskType.MULTICLASS_CLASSIFICATION:
        # assert that all features are numerical
        assert (len(out.col_names_dict[stype.numerical]) == total_cols)
        assert (len(
            dataset.tensor_frame.col_names_dict[stype.categorical]) == len(
                out.col_names_dict[stype.numerical][total_numerical_cols:]))

        # The first categorical column, x, is changed to all 0's initially.
        # This assert statement makes sure that transform is correct.
        # The transform uses m-target estimation, where each categorical
        # feature is transformed to (num_count + target_mean)/(num_rows + 1).
        # This test tests the correctness in multiclass classification task.
        assert torch.allclose(
            out.feat_dict[stype.numerical][:, total_numerical_cols].float(),
            torch.tensor((num_rows + dataset.tensor_frame.y.float().mean()) /
                         (num_rows + 1), device=out.device).repeat(num_rows))
    else:
        # when the task is multiclass classification, the number of
        # columns changes.
        assert (len(out.col_names_dict[stype.numerical]) == (
            total_numerical_cols + (dataset.num_classes - 1) *
            (total_cols - total_numerical_cols)))

    # assert that the numerical features are unchanged
    assert (torch.eq(
        dataset.tensor_frame.feat_dict[stype.numerical],
        out.feat_dict[stype.numerical][:, :total_numerical_cols]).all())
    assert (dataset.tensor_frame.col_names_dict[stype.numerical] ==
            out.col_names_dict[stype.numerical][:total_numerical_cols])
