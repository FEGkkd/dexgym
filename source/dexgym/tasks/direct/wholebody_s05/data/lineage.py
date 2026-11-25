"""Exact-copy canonicalization and explicitly evidenced ancestry."""
from collections import defaultdict


def exact_aliases(rows):
    """Return duplicate ID -> canonical ID, using complete signatures only."""
    by_signature = defaultdict(list)
    for row in rows:
        if row['signature']:
            by_signature[row['signature']].append(row)
    aliases = {}
    for group in by_signature.values():
        ranked = sorted(group, key=lambda row: (-row['priority'], row['id']))
        aliases.update({row['id']: ranked[0]['id'] for row in ranked[1:]})
    return aliases


class Lineage:
    def __init__(self, parents):
        self.parents = {key: tuple(sorted(set(values))) for key, values in parents.items()}
        self._resolved = {}

    def resolve(self, node):
        """Return all resolved roots and all unresolved ancestors, independently."""
        return self._walk(node, set())

    def _walk(self, node, active):
        if node in self._resolved:
            return self._resolved[node]
        if node not in self.parents:
            return (), (node,)
        if node in active:
            raise ValueError(f'lineage cycle at {node}')
        parents = self.parents[node]
        if not parents:
            value = (node,), ()
        else:
            roots, missing = set(), set()
            active.add(node)
            for parent in parents:
                found, unresolved = self._walk(parent, active)
                roots.update(found)
                missing.update(unresolved)
            active.remove(node)
            value = tuple(sorted(roots)), tuple(sorted(missing))
        self._resolved[node] = value
        return value
