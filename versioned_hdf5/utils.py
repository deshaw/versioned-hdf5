from typing import Dict

import h5py
import numpy as np
from ndindex import Slice, Tuple

from .hashtable import Hashtable
from .slicetools import partition


class Chunk:
    pass


class AppendChunk(Chunk):
    def __init__(
        self,
        target_raw_index: Tuple,
        target_raw_data: np.ndarray,
        raw_last_chunk: Tuple,
        raw_last_chunk_data: np.ndarray,
    ):
        """Instantiate an AppendChunk.

        Parameters
        ----------
        target_raw_index : Tuple
            Indices in rspace where target_raw_data is to be written
        target_raw_data : np.ndarray
            Raw data to write (this is only the part being appended)
        raw_last_chunk : Tuple
            Index of the last chunk of the raw_data post-append
        raw_last_chunk_data : np.ndarray
            Data of the last chunk of the raw_data post-append
        """
        # Check the hash of this before writing to the raw data
        self.new_raw_last_chunk_data = raw_last_chunk_data
        self.new_raw_last_chunk = raw_last_chunk

        # Write this after checking the hash
        self.target_raw_index = target_raw_index
        self.target_raw_data = target_raw_data

    def write_to_raw(self, hashtable: Hashtable, raw_data: h5py.Dataset) -> Tuple:
        # If the hash of the data is already in the hash table,
        # just reuse the hashed slice. Otherwise, update the
        # hash table with the new data hash and write the data
        # to the raw dataset.
        data_hash = hashtable.hash(self.new_raw_last_chunk_data)
        if data_hash in hashtable:
            return hashtable[data_hash]

        # Update the hashtable
        hashtable[data_hash] = self.new_raw_last_chunk_data

        # Write only the data to append
        raw_data[self.target_raw_index.raw] = self.target_raw_data

        # Keep track of the last index written to in the raw dataset;
        # future appends are simplified by this
        raw_data.attrs["last_element"] = self.rchunk.args[0].stop

        # Return the last raw chunk
        return self.new_raw_last_chunk_data


class WriteChunk(Chunk):
    def __init__(self, vchunk: Tuple, data: np.ndarray):
        self.vchunk = vchunk
        self.data = data

    def write_to_raw(
        self, hashtable: Hashtable, raw_data: h5py.Dataset
    ) -> Dict[Tuple, Tuple]:
        chunk_size = raw_data.chunks[0]

        if isinstance(self.data, np.ndarray) and raw_data.dtype != self.data.dtype:
            raise ValueError(
                f"dtype of raw data ({raw_data.dtype}) does not match data to append "
                f"({self.data.dtype})"
            )

        chunks: Dict[Tuple, Tuple] = {}
        for data_slice, vchunk in zip(
            partition(self.data, raw_data.chunks),
            partition(self.vchunk, raw_data.chunks),
        ):
            arr = self.data[data_slice.raw]
            data_hash = hashtable.hash(arr)

            if data_hash in hashtable:
                chunks[vchunk] = hashtable[data_hash]
            else:
                new_chunk_axis_size = raw_data.shape[0] + len(data_slice.args[0])

                rchunk = Tuple(
                    Slice(
                        raw_data.shape[0],
                        new_chunk_axis_size,
                    ),
                    *[Slice(None, None) for _ in raw_data.shape[1:]],
                )

                # Resize the dataset to include a new chunk
                raw_data.resize(raw_data.shape[0] + chunk_size, axis=0)

                # Map the virtual chunk to the raw data chunk
                chunks[vchunk] = rchunk

                # Map the data hash to the raw data chunk
                hashtable[data_hash] = rchunk

                # Set the value of the raw data chunk to the chunk of the new data being written
                raw_data[rchunk.raw] = self.data[data_slice.raw]

                # Update the last element attribute of the raw dataset
                raw_data.attrs["last_element"] = rchunk.args[0].stop

        return chunks
