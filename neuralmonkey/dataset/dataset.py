"""Implementation of the dataset class."""
import random
import collections
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from typeguard import check_argument_types


class Dataset(collections.Sized):
    """Base Dataset class.

    This class serves as collection for data series for particular
    encoders and decoders in the model. If it is not provided a parent
    dataset, it also manages the vocabularies inferred from the data.

    A data series is either a list of strings or a numpy array.
    """

    def __init__(self,
                 name: str, series: Dict[str, List],
                 series_outputs: Dict[str, str],
                 primary_series: str = None,
                 preprocessors: List[Tuple[str, str, Callable]] = None
                ) -> None:
        """Create a dataset from the provided series of data.

        Arguments:
            name: The name for the dataset
            series: Dictionary from the series name to the actual data.
            series_outputs: Output files for target series.
            primary_series: Reference series that is used for bucketing.
            preprocessors: The definition of the preprocessors.
        """
        check_argument_types()

        self.name = name
        self._series = series
        self.series_outputs = series_outputs
        self._primary_series = primary_series

        if self._primary_series is None:
            self._primary_series = list(self._series.keys())[0]

        if preprocessors is not None:
            for src_id, tgt_id, function in preprocessors:
                if src_id == tgt_id:
                    raise Exception(
                        "Attempt to rewrite series '{}'".format(src_id))
                if src_id not in self._series:
                    raise Exception(
                        ("The source series ({}) of the '{}' preprocessor "
                         "is not defined in the dataset.").format(
                             src_id, str(function)))
                self._series[tgt_id] = [
                    function(item) for item in self._series[src_id]]

        self._check_series_lengths()

    def _check_series_lengths(self) -> None:
        """Check lenghts of series in the dataset.

        Raises:
            ValueError when the lengths in the dataset do not match.
        """
        lengths = [len(list(v)) for v in self._series.values()
                   if isinstance(v, (list, np.ndarray))]

        if len(set(lengths)) > 1:
            err_str = ["{}: {}".format(s, len(list(self._series[s])))
                       for s in self._series]
            raise ValueError("Lengths of data series must be equal. Were: {}"
                             .format(", ".join(err_str)))

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            The length of the dataset.
        """
        if not list(self._series.values()):
            return 0

        first_series = next(iter(self._series.values()))
        return len(list(first_series))

    def has_series(self, name: str) -> bool:
        """Check if the dataset contains a series of a given name.

        Arguments:
            name: Series name

        Returns:
            True if the dataset contains the series, False otherwise.
        """
        return name in self._series

    def maybe_get_series(self, name: str) -> Optional[Iterable]:
        """Get the data series with a given name.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            The data series or None if it does not exist.
        """
        return self._series.get(name, None)

    def get_series(self, name: str) -> Iterable:
        """Get the data series with a given name.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            The data series.

        Raises:
            KeyError if the series does not exists.
        """
        return self._series[name]

    @property
    def series_ids(self) -> Iterable[str]:
        return self._series.keys()

    def shuffle(self) -> None:
        """Shuffle the dataset randomly."""
        keys = list(self._series.keys())
        zipped = list(zip(*[self._series[k] for k in keys]))
        random.shuffle(zipped)
        for key, serie in zip(keys, list(zip(*zipped))):
            self._series[key] = serie

    # pylint: disable=no-self-use
    def _is_over_batch_size(self,
                            batch: List[List[str]],
                            batch_size: int,
                            token_level: bool) -> bool:
        size = len(batch)
        if token_level:
            size = sum([len(x[0]) for x in batch])
        return size >= batch_size
    # pylint: enable=no-self-use

    def batch_series(self,
                     series_names: List[str],
                     batch_size: int,
                     bucket_span: int,
                     token_level: bool) -> Iterable[Iterable]:
        """Split data series into batches.

        Entries are "bucketed" so that the entries of the similar size
        are in the same batch.

        Arguments:
            series_names: List of the names of the series.
            batch_size: The size of a batch.
            bucket_span: A span that is covered by each bucket.
            token_level: Count the batch_size as number of tokens.

        Returns:
            Generator yielding batches of the data from the series.
        """
        buckets = {}  # type: Dict[int, List[Any]]
        for items in zip(*[self.get_series(key) for key in series_names]):
            if bucket_span == -1:
                bucket_id = 0
            else:
                bucket_id = len(items[0]) // bucket_span
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(items)
            if self._is_over_batch_size(buckets[bucket_id],
                                        batch_size, token_level):
                yield [list(x) for x in zip(*buckets[bucket_id])]
                buckets[bucket_id] = []

        for bucket_id in buckets:
            if buckets[bucket_id]:
                yield [list(x) for x in zip(*buckets[bucket_id])]

    def batch_dataset(self,
                      batch_size: int,
                      bucket_span: int,
                      token_level: bool) -> Iterable["Dataset"]:
        """Split the dataset into a list of batched datasets.

        Arguments:
            batch_size: The size of a batch.
            bucket_span: The span of buckets for bucketed batching.
                Default value -1 means no bucketing.

        Returns:
            Generator yielding batched datasets.
        """
        keys = list(self._series.keys())
        keys.insert(0, keys.pop(keys.index(self._primary_series)))
        batched_series = self.batch_series(
            keys, batch_size, bucket_span, token_level)

        batch_index = 0
        for next_batches in batched_series:
            batch_dict = {key: data for key, data in zip(keys, next_batches)}
            dataset = Dataset(self.name + "-batch-{}".format(batch_index),
                              batch_dict, {})
            batch_index += 1
            yield dataset

    def add_series(self, name: str, series: List) -> None:
        if name in self._series:
            raise ValueError(
                "Can't add series that already exist: {}".format(name))
        self._series[name] = series

    def subset(self, start: int, length: int) -> "Dataset":
        subset_name = "{}.{}.{}".format(self.name, start, length)
        subset_outputs = {k: "{}.{:010}".format(v, start)
                          for k, v in self.series_outputs.items()}
        subset_series = {k: v[start:start + length]
                         for k, v in self._series.items()}

        return Dataset(subset_name, subset_series, subset_outputs)
