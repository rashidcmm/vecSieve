from __future__ import annotations


class IdMap:
    """External id <-> internal dense row index, assigned sequentially on add()."""

    def __init__(self):
        self._ext_to_int: dict = {}
        self._int_to_ext: list = []

    def add(self, external_id) -> int:
        if external_id in self._ext_to_int:
            raise ValueError(f"duplicate external id: {external_id!r}")
        internal = len(self._int_to_ext)
        self._ext_to_int[external_id] = internal
        self._int_to_ext.append(external_id)
        return internal

    def to_internal(self, external_id) -> int:
        return self._ext_to_int[external_id]

    def to_external(self, internal_row: int):
        return self._int_to_ext[internal_row]

    def __len__(self) -> int:
        return len(self._int_to_ext)
