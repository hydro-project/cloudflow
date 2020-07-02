from typing import Any, Dict, List, Tuple

import cloudpickle as cp
import numpy as np
import pyarrow as pa

from flow.types.basic import get_from_proto, Type, NumpyType, BASIC_TYPES
from flow.types.error import FlowError
from flow.types.flow_pb2 import (
    ProtoSchema,
    ProtoTable
)

class Row:
    qid_key = 'qid'

    def __init__(self, schema: List[Tuple[str, Type]], vals: List[Any], qid:
                 int):
        self.data = {}
        self.schema = schema

        for idx, val in enumerate(vals):
            name, _ = schema[idx]
            self.data[name] = val

        self.data[Row.qid_key] = qid

    def clone(self):
        vals = []
        for key, _ in self.schema:
            vals.append(self.data[key])

        new_row = Row(self.schema, vals, self.data[Row.qid_key])
        return new_row

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key == Row.qid_key:
            raise FlowError('Cannot modify query ID of a row.')

        for name, tp in self.schema:
            if key == name:
                if not isinstance(value, tp.typ):
                    raise FlowError(f'Invalid update to {key}: Does not match'
                                    + f'type {tp}.')
                else:
                    break

        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def __str__(self):
        vals = []
        for name, _ in self.schema:
            data = self.data[name]
            vals.append(name + ': ' + str(data))

        return '[' + ', '.join(vals) + ']'

def merge_schema(lschema, rschema):
    new_schema = []
    new_schema += lschema

    for tp in rschema:
        if tp not in new_schema:
            new_schema.append(tp)

    return new_schema

def merge_row(left: Row, right: Row, schema):
    vals = []
    for fld, _ in schema:
        if fld in left:
            vals.append(left[fld])
        else:
            vals.append(right[fld])

    if left[Row.qid_key] == right[Row.qid_key]:
        qid = left[Row.qid_key]
    else:
        qid = left[Row.qid_key] + right[Row.qid_key]

    return Row(schema, vals, qid=qid)


class Table:
    def __init__(self, schema: List[Tuple[str, Type]]):
        self.schema = schema
        self.data = []
        self.qid_count = 0

        for _, typ in schema:
            if typ not in BASIC_TYPES:
                raise RuntimeError(f'{typ.__name__} is not a valid column type.')

    def _check(self, vals: List[Any]):
        if len(vals) != len(self.schema):
            raise FlowError(f'Expected {len(self.schema)} vals but found'
                            + f' {len(vals)}')

        for idx, val in enumerate(vals):
            name, typ = self.schema[idx]

            # None values are okay because they are NULLs.
            if val is not None and not isinstance(val, typ.typ):
                raise FlowError(f'Expected type {typ.typ} but instead found' +
                                f' {type(val)} for column {name}')

        return True

    def insert(self, vals: List[Any], qid: int=None) -> bool:
        if isinstance(vals, Row):
            if vals.schema == self.schema:
                self.data.append(vals)
                return True

            raise FlowError(f'Invalid row insertion: {vals}')

        if not isinstance(vals, list):
            raise FlowError('Unrecognized type: ' + str(vals) +
                            '\n\nCan only insert a Row or a list.')


        if not self._check(vals):
            return False

        if qid is None:
            self.data.append(Row(self.schema, vals, self.qid_count))
            self.qid_count += 1
        else:
            self.data.append(Row(self.schema, vals, qid))

        return True

    def get(self):
        for row in self.data:
            yield row

    def size(self):
        return len(self.data)

    def clone(self):
        new_table = Table(self.schema)

        for row in self.data:
            new_row = row.clone()
            new_table.insert(new_row)

        return new_table

    def __str__(self):
        return '\n'.join([str(row) for row in self.data])


class GroupbyTable:
    def __init__(self, schema: List[Tuple[str, Type]], col: str):
        self.col = col
        self.schema = schema

        self.tables = {}

    def add_row(self, row: Row):
        key = row[self.col]
        if key not in self.tables:
            self.tables[key] = Table(self.schema)

        self.tables[key].insert(row)

    def add_group(self, val, table: Table):
        self.tables[val] = table

    def get(self):
        for group in self.tables:
            yield (group, self.tables[group])


def merge_tables(tables: List[Table]) -> (Table, Dict[int, int]):
    schema = tables[0].schema
    for other in tables[1:]:
        if other.schema != schema:
            raise FlowError(f'Schema mismatch:\n\t{schema}\n\t{other.schema}')

    mappings = {}
    mappings['NUM_TABLES'] = len(tables)
    result = Table(schema)

    qid = 0
    for idx, table in enumerate(tables):
        for row in table.get():
            # Convert to a list, so it gets assigned a new qid.
            vals = []
            for val, _ in schema:
                vals.append(row[val])

            result.insert(vals, qid)
            mappings[qid] = idx
            qid += 1

    return result, mappings

def demux_tables(table: Table, mappings: Dict[int, int]):
    result = []
    num_tables = mappings['NUM_TABLES']
    for _ in range(num_tables):
        tbl = Table(table.schema)
        result.append(tbl)

    for row in table.get():
        # Convert to a list, so query ID get reset.
        vals = []
        for val, _ in table.schema:
            vals.append(row[val])

        qid = row[Row.qid_key]
        tbl_index = mappings[qid]
        print(f'Trying to insert at {tbl_index} and there {len(result)} tables')

        result[tbl_index].insert(vals)

    return result

def serialize(table: Table) -> Tuple:
    import time
    start = time.time()

    result = ProtoTable()
    pschema = result.schema
    for name, tp in table.schema:
        col = pschema.columns.add()
        col.name = name
        col.type = tp.proto

    for row in table.get():
        prow = result.rows.add()
        for col, tp in table.schema:
            val = row[col]
            if tp == NumpyType:
                ser = pa.serialize(val).to_buffer().to_pybytes()
            else:
                ser = cp.dumps(val)

            prow.values.append(ser)

        prow.qid = row[Row.qid_key]

    return result.SerializeToString()

    # table = table.clone()
    # numpy_cols = list(map(lambda col: col[0], filter(lambda col: col[1] ==
    #                                                  NumpyType, table.schema)))
    # result = [[]] * (1 + len(numpy_cols))

    # empty = np.array([])
    # row = table.data[0]

    # merged = []
    # for col in numpy_cols:
    #     all_for_col = []
    #     for row in table.get():
    #         all_for_col.append(row[col])

    #     merged.append(np.hstack(all_for_col))

    # result[0] = table
    # for idx, arr in enumerate(merged):
    #     result[idx + 1] = pa.serialize(arr).to_buffer().to_pybytes()

    # print(b'0xa' in result[1])
    # end = time.time()
    # print(f'Total {end - start}')
    # return tuple(result)

def deserialize(serialized: bytes) -> Table:
    ptable = ProtoTable()
    ptable.ParseFromString(serialized)

    schema = []
    for col in ptable.schema.columns:
        schema.append((col.name, get_from_proto(col.type)))

    table = Table(schema)

    for row in ptable.rows:
        vals = []
        for idx, val in enumerate(row.values):
            if schema[idx][1] == NumpyType:
                val = pa.deserialize(val)
            else:
                val = cp.loads(val)

            vals.append(val)

        table.insert(vals)

    return table
