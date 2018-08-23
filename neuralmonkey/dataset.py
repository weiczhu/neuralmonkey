"""Implementation of the dataset class."""

import os
import random
import re
import glob
import collections
from itertools import islice

from typing import cast, Any, List, Callable, Iterable, Dict, Tuple, Union

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.logging import log, debug
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader

# pylint: disable=invalid-name
Reader = Callable[[List[str]], Any]
ReaderDef = Union[str, List[str],
                  Tuple[str, Reader], Tuple[List[str], Reader]]
DatasetPreprocess = Callable[["Dataset"], Iterable[Any]]
DatasetPostprocess = Callable[["Dataset", Dict[str, Iterable[Any]]],
                              Iterable[Any]]
SeriesConfig = Dict[str, Union[ReaderDef, DatasetPreprocess]]
# pylint: enable=invalid-name

SERIES_SOURCE = re.compile("s_([^_]*)$")
SERIES_OUTPUT = re.compile("s_(.*)_out")
PREPROCESSED_SERIES = re.compile("pre_([^_]*)$")


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
                 preprocessors: List[Tuple[str, str, Callable]] = None
                ) -> None:
        """Create a dataset from the provided series of data.

        Arguments:
            name: The name for the dataset
            series: Dictionary from the series name to the actual data.
            series_outputs: Output files for target series.
            preprocessors: The definition of the preprocessors.
        """
        self.name = name
        self._series = dict(series)
        self.series_outputs = series_outputs

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
            Exception when the lengths in the dataset do not match.
        """
        lengths = [len(list(v)) for v in self._series.values()
                   if isinstance(v, (list, np.ndarray))]

        if len(set(lengths)) > 1:
            err_str = ["{}: {}".format(s, len(list(self._series[s])))
                       for s in self._series]
            raise Exception("Lengths of data series must be equal. Instead: {}"
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

    def get_series(self, name: str, allow_none: bool = False) -> Iterable:
        """Get the data series with a given name.

        Arguments:
            name: The name of the series to fetch.
            allow_none: If True, return None if the series does not exist.

        Returns:
            The data series.

        Raises:
            KeyError if the series does not exists and allow_none is False
        """
        if allow_none:
            return self._series.get(name)

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

    def batch_serie(self, serie_name: str,
                    batch_size: int) -> Iterable[Iterable]:
        """Split a data serie into batches.

        Arguments:
            serie_name: The name of the series
            batch_size: The size of a batch

        Returns:
            Generator yielding batches of the data from the serie.
        """
        buf = []
        for item in self.get_series(serie_name):
            buf.append(item)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def batch_bucketed(self, series_names: List[str],
                       batch_size: int) -> Iterable[Iterable]:
        buckets = {}  # type: Dict[int, List[Any]]
        for items in zip(*[self.get_series(key) for key in series_names]):
            bucket_id = max(len(i) for i in items) % 5
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(items)
            if len(buckets[bucket_id]) > batch_size:
                yield zip(*buckets[bucket_id])
                buckets[bucket_id] = []

        for bucket_id in buckets:
            if buckets[bucket_id]:
                yield zip(*buckets[bucket_id])

    def batch_dataset(self,
                      batch_size: int,
                      batching_scheme: str) -> Iterable["Dataset"]:
        """Split the dataset into a list of batched datasets.

        Arguments:
            batch_size: The size of a batch.

        Returns:
            Generator yielding batched datasets.
        """
        keys = list(self._series.keys())
        if batching_scheme == "basic":
            batched_series = [self.batch_serie(key, batch_size)
                              for key in keys]
        elif batching_scheme == "bucketed":
            batched_series = [self.batch_bucketed(keys, batch_size)]

        batch_index = 0
        for next_batches in zip(*batched_series):
            batch_dict = {key: data for key, data in zip(keys, next_batches)}
            dataset = Dataset(self.name + "-batch-{}".format(batch_index),
                              batch_dict, {})
            batch_index += 1
            yield dataset

    def add_series(self, name: str, series: List[Any]) -> None:
        if name in self._series:
            raise ValueError(
                "Can't series that already exist: {}".format(name))
        self._series[name] = series

    def subset(self, start: int, length: int) -> "Dataset":
        subset_name = "{}.{}.{}".format(self.name, start, length)
        subset_outputs = {k: "{}.{:010}".format(v, start)
                          for k, v in self.series_outputs.items()}
        subset_series = {k: v[start:start + length]
                         for k, v in self._series.items()}

        return Dataset(subset_name, subset_series, subset_outputs)


class LazyDataset(Dataset):
    """Implements the lazy dataset.

    The main difference between this implementation and the default one is
    that the contents of the file are not fully loaded to the memory.
    Instead, everytime the function ``get_series`` is called, a new file handle
    is created and a generator which yields lines from the file is returned.
    """

    def __init__(self, name: str,
                 series_paths_and_readers: Dict[str, Tuple[List[str], Reader]],
                 series_outputs: Dict[str, str],
                 preprocessors: List[Tuple[str, str, Callable]] = None
                ) -> None:
        """Create a new instance of the lazy dataset.

        Arguments:
            name: The name of the dataset series_paths_and_readers: The mapping
            of series name to its file series_outputs: Dictionary mapping
            series names to their output file preprocess: The preprocessor to
            apply to the read lines
        """
        parent_series = dict()  # type: Dict[str, Any]
        parent_series.update({s: None for s in series_paths_and_readers})
        if preprocessors:
            parent_series.update({s[1]: None for s in preprocessors})
        super().__init__(name, parent_series, series_outputs)
        self.series_paths_and_readers = series_paths_and_readers

        for series_name, (paths, _) in series_paths_and_readers.items():
            for path in paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        "File not found. Series: {}, Path: {}"
                        .format(series_name, path))

        self.preprocess_series = {}  # type: Dict[str, Tuple[str, Callable]]
        if preprocessors is not None:
            for src_id, tgt_id, func in preprocessors:
                if src_id == tgt_id:
                    raise Exception(
                        "Attempt to rewrite series '{}'".format(src_id))
                if src_id not in series_paths_and_readers:
                    raise Exception(
                        ("The source series ({}) of the '{}' preprocessor "
                         "is not defined in the dataset.").format(
                             src_id, str(func)))
                self.preprocess_series[tgt_id] = (src_id, func)

    def has_series(self, name: str) -> bool:
        """Check if the dataset contains a series of a given name.

        Arguments:
            name: Series name

        Returns:
            True if the dataset contains the series, False otherwise.
        """
        return (name in self.series_paths_and_readers
                or name in self.preprocess_series)

    def get_series(self, name: str, allow_none: bool = False) -> Iterable:
        """Get the data series with a given name.

        This function opens a new file handle and returns a generator which
        yields preprocessed lines from the file.

        Arguments:
            name: The name of the series to fetch.
            allow_none: If True, return None if the series does not exist.

        Returns:
            The data series.

        Raises:
            KeyError if the series does not exists and allow_none is False
        """
        if (allow_none
                and name not in self.series_paths_and_readers
                and name not in self.preprocess_series):
            return None

        if name in self.series_paths_and_readers:
            paths, reader = self.series_paths_and_readers[name]
            return reader(paths)
        elif name in self.preprocess_series:
            src_id, func = self.preprocess_series[name]
            src_series = self.get_series(src_id, allow_none)
            if src_series is None:
                return None
            return (func(item) for item in src_series)
        else:
            raise Exception("Series '{}' is not in the dataset.".format(name))

    def shuffle(self) -> None:
        """Do nothing, not in-memory shuffle is impossible.

        TODO: this is related to the ``__len__`` method.
        """
        pass

    @property
    def series_ids(self) -> Iterable[str]:
        return (list(self.series_paths_and_readers.keys())
                + list(self.preprocess_series.keys()))

    def add_series(self, name: str, series: Iterable[Any]) -> None:
        raise NotImplementedError(
            "Lazy dataset does not support adding series.")

    def subset(self, start: int, length: int) -> Dataset:
        subset_name = "{}.{}.{}".format(self.name, start, length)
        subset_outputs = {k: "{}.{:010}".format(v, start)
                          for k, v in self.series_outputs.items()}

        # TODO make this more efficient with large datasets
        subset_series = {
            s_id: list(islice(self.get_series(s_id), start, start + length))
            for s_id in self.series_ids}

        return Dataset(subset_name, subset_series, subset_outputs)


def from_files(
        name: str = None, lazy: bool = False,
        preprocessors: List[Tuple[str, str, Callable]] = None,
        **kwargs) -> Dataset:
    """Load a dataset from the files specified by the provided arguments.

    Paths to the data are provided in a form of dictionary.

    Keyword arguments:
        name: The name of the dataset to use. If None (default), the name will
              be inferred from the file names.
        lazy: Boolean flag specifying whether to use lazy loading (useful for
              large files). Note that the lazy dataset cannot be shuffled.
              Defaults to False.
        preprocessor: A callable used for preprocessing of the input sentences.
        kwargs: Dataset keyword argument specs. These parameters should begin
                with 's_' prefix and may end with '_out' suffix.  For example,
                a data series 'source' which specify the source sentences
                should be initialized with the 's_source' parameter, which
                specifies the path and optinally reader of the source file. If
                runners generate data of the 'target' series, the output file
                should be initialized with the 's_target_out' parameter.
                Series identifiers should not contain underscores.
                Dataset-level preprocessors are defined with 'pre_' prefix
                followed by a new series name. In case of the pre-processed
                series, a callable taking the dataset and returning a new
                series is expected as a value.

    Returns:
        The newly created dataset.

    Raises:
        Exception when no input files are provided.
    """
    check_argument_types()

    series_paths_and_readers = _get_series_paths_and_readers(kwargs)
    series_outputs = _get_series_outputs(kwargs)

    if not series_paths_and_readers:
        raise Exception("No input files are provided.")

    log("Initializing dataset with: {}".format(
        ", ".join(series_paths_and_readers)))

    if lazy:
        dataset = LazyDataset(name, series_paths_and_readers, series_outputs,
                              preprocessors)  # type: Dataset
    else:
        series = {key: list(reader(paths))
                  for key, (paths, reader) in series_paths_and_readers.items()}

        dataset = Dataset(name, series, series_outputs, preprocessors)
        log("Dataset length: {}".format(len(dataset)))

    _preprocessed_datasets(dataset, kwargs)

    return dataset


load_dataset_from_files = from_files  # pylint: disable=invalid-name


def _get_series_paths_and_readers(
        series_config: SeriesConfig) -> Dict[str, Tuple[List[str], Reader]]:
    """Get paths to files that contain data from the dataset kwargs.

    Input file for a serie named 'xxx' is specified by parameter 's_xxx'. The
    dataset series is defined by a string with a path / list of strings with
    paths, or a tuple whose first member is a path or a list of paths and the
    second memeber is a reader function.

    The paths can contain wildcards, which will be expanded using
    :py:func:`glob.glob` in sorted order.

    Arguments:
        series_config: A dictionary containing the dataset keyword argument
            specs.

    Returns:
        A dictionary which maps serie names to the paths of their input files
        and readers..
    """
    keys = [k for k in list(series_config.keys()) if SERIES_SOURCE.match(k)]
    names = [SERIES_SOURCE.match(k).group(1) for k in keys]

    series_sources = {}
    for name, key in zip(names, keys):
        value = cast(ReaderDef, series_config[key])

        if isinstance(value, tuple):
            patterns, reader = value  # type: ignore
        else:
            patterns = value
            reader = UtfPlainTextReader

        if isinstance(patterns, str):
            patterns = [patterns]

        paths = []
        for pattern in patterns:
            matched_files = sorted(glob.glob(pattern))
            if not matched_files:
                raise FileNotFoundError(
                    "Pattern did not match any files. Series: {}, Pattern: {}"
                    .format(name, pattern))
            paths.extend(matched_files)

        debug("Series '{}' has the following files: {}".format(name, paths))

        series_sources[name] = (paths, reader)

    return series_sources


def _get_series_outputs(series_config: SeriesConfig) -> Dict[str, str]:
    """Get paths to series outputs from the dataset keyword argument specs.

    Output file for a series named 'xxx' is specified by parameter 's_xxx_out'

    Arguments:
        series_config: A dictionary containing the dataset keyword argument
           specs.

    Returns:
        A dictionary which maps serie names to the paths for their output
        files.
    """
    outputs = {}
    for key, value in series_config.items():
        matcher = SERIES_OUTPUT.match(key)
        if matcher:
            name = matcher.group(1)
            if not isinstance(value, str):
                raise ValueError(
                    "Output path for '{}' series must be a string, was {}.".
                    format(name, type(value)))
            outputs[name] = cast(str, value)
    return outputs


def _preprocessed_datasets(
        dataset: Dataset,
        series_config: SeriesConfig) -> None:
    """Apply dataset-level preprocessing."""
    keys = [key for key in series_config.keys()
            if PREPROCESSED_SERIES.match(key)]

    for key in keys:
        name = PREPROCESSED_SERIES.match(key).group(1)
        preprocessor = cast(DatasetPreprocess, series_config[key])

        if isinstance(dataset, Dataset):
            new_series = list(preprocessor(dataset))
            dataset.add_series(name, new_series)
        elif isinstance(dataset, LazyDataset):
            dataset.preprocess_series[name] = (None, preprocessor)
