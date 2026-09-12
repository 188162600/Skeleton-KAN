"""Frozen catalogue validation and materialization, separate from neural layers."""
import copy
import hashlib
import json
from pathlib import Path

class Catalogue:
    """A finite collection of explicit composition specifications.

    This layer loads/freezes constructor outputs. It is not a substitute for
    a published clustering algorithm and does not invent target-informed G.
    """

    def __init__(self, entries):
        self.entries = copy.deepcopy(list(entries))
        if not self.entries:
            raise ValueError("a catalogue cannot be empty")
        ids, structures = set(), set()
        for entry in self.entries:
            if set(entry) != {'id','structure'} or not isinstance(entry['id'],str) or not entry['id']:
                raise ValueError("each entry must contain a nonempty id and a structure")
            encoded = json.dumps(entry['structure'],sort_keys=True,separators=(',',':'))
            if entry['id'] in ids or encoded in structures:
                raise ValueError("duplicate catalogue id or exact duplicate specification")
            ids.add(entry['id']);structures.add(encoded)

    @property
    def sha256(self):
        payload = json.dumps(self.entries,sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def save(self, path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(self.entries,indent=2),encoding='utf-8')

    def materialize(self, input_dim, *, train_inputs=None, **model_options):
        from ..model import StructuredKANBuilder
        constructor = StructuredKANBuilder(input_dim,**model_options)
        return {entry['id']:constructor.build(entry['structure'],train_inputs=train_inputs) for entry in self.entries}
