"""Native ACUOS2 adapter; inference code remains in the authors Maude files."""
from pathlib import Path
import hashlib
import json
import sys
import time
try:
    import resource
except ImportError:
    resource = None

_PAIR_BOOTSTRAP = None


def peak_rss_mib():
    """Report peak resident memory where supported; missing is not zero."""
    if resource is None:
        return None
    divisor = 1024 * 1024 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def pair_bootstrap():
    if _PAIR_BOOTSTRAP is None:
        raise RuntimeError('Configure the ACUOS2 signature before starting a pair process.')
    return dict(_PAIR_BOOTSTRAP)


def restore_pair_modules(bootstrap):
    """Reconstruct native modules in a spawned child, without pickling Maude objects."""
    maude, _ = load(bootstrap['vendor'])
    assert maude.input(bootstrap['specification'])
    return maude.getModule('TEST'), maude.getModule('SPECIFICATION')
ARCHIVE_SHA256="d9214e03f135dc0798d198bbfea48b4b4fdab6d0e51322b13d74b3b402d311e6"
FILES=["meta-ext","lgg-general","lgg-lub-sorts","lgg-decompose-nax","lgg-decompose-a","lgg-decompose-c","lgg-decompose-ac","lgg-decompose-au","lgg-decompose-acu","lgg-rules","acuos2"]

def load(vendor):
    import maude
    maude.init(advise=False)
    folder = Path(vendor) / 'acuos2-source'
    hashes = {}
    changes = []
    for name in FILES:
        path = folder / (name + '.maude')
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        source = raw.decode('utf-8')
        source = '\n'.join(line for line in source.splitlines()
                           if not line.strip().startswith('load '))
        n = source.count('(none).EmptyTypeSet')
        if n:
            changes.append(dict(file=path.name, old='(none).EmptyTypeSet',
                                new='(none).TypeSet', occurrences=n))
            source = source.replace('(none).EmptyTypeSet', '(none).TypeSet')
        assert maude.input(source), path.name
    assert maude.getModule('ACUOS2') is not None
    return maude, dict(archive_sha256=ARCHIVE_SHA256, native_file_sha256=hashes,
                       loader_compatibility=changes, inference_rules_changed=False)


def configure(trees, vendor, theory='ACU'):
    global _PAIR_BOOTSTRAP
    maude, provenance = load(vendor)
    arities = {}
    def collect(t):
        if t.name not in ('sum', 'prod'):
            if t.name in arities:
                assert arities[t.name] == len(t.children), t.name
            arities[t.name] = len(t.children)
        for child in t.children:
            collect(child)
    for tree in trees:
        collect(tree)
    tokens = {name: f'o{i}' for i, name in enumerate(sorted(arities))}
    tokens.update({'sum': 's', 'prod': 'p', 'constant:0': 'e0', 'constant:1': 'e1'})
    assert theory in ('AC', 'ACU')
    decl = ['ops e0 e1 : -> E .',
            'op s : E E -> E [assoc comm' + (' id: e0' if theory == 'ACU' else '') + '] .',
            'op p : E E -> E [assoc comm' + (' id: e1' if theory == 'ACU' else '') + '] .']
    for name, n in sorted(arities.items()):
        if name not in ('constant:0', 'constant:1'):
            decl.append('op ' + tokens[name] + ' : ' + ' '.join(['E'] * n) + ' -> E .')
    specification = ('fmod SPECIFICATION is sort E . ' + ' '.join(decl) +
                     ' endfm fmod TEST is inc ACUOS2 + SPECIFICATION . endfm')
    assert maude.input(specification)
    _PAIR_BOOTSTRAP = {'vendor': str(Path(vendor).resolve()), 'specification': specification}
    return maude.getModule('TEST'), maude.getModule('SPECIFICATION'), tokens, provenance


def matches(subject, pattern):
    return next(iter(subject.match(pattern, maxDepth=-1)), None) is not None


def generalize(test, spec, left, right, entry='auto'):
    start = time.perf_counter()
    a, b = spec.parseTerm(left), spec.parseTerm(right)
    assert a is not None and b is not None, (left, right)
    if entry == 'auto':
        # Keep the normal published entry except its two defective atomic
        # branches. Normalize in the original object theory before dispatch.
        a.reduce()
        b.reduce()
        left, right = str(a), str(b)
        atomic_pair = not list(a.arguments()) and not list(b.arguments())
        entry = 'constraint' if atomic_pair and left != right else 'standard'
    if entry == 'constraint':
        # Native overload used internally by eq&lggs4, valid for our single
        # E sort. This avoids the archive's defective atom-list accumulator
        # without changing any published equation/inference/filter rule.
        assert 'X#:' not in left and 'X#:' not in right
        expression = "lggs(fixModuleForLGG(upModule('SPECIFICATION,true)), {upTerm(" + left + ") $ 'X#:E $ upTerm(" + right + ')})'
    else:
        expression = "lggs(upModule('SPECIFICATION, true), upTerm(" + left + '), upTerm(' + right + '))'
    term = test.parseTerm(expression)
    assert term is not None
    rewrites = term.reduce()  # Complete native set, not a first-solution search.
    seconds = time.perf_counter() - start
    native = str(term)
    pieces = list(term.arguments()) if str(term.symbol()) == '_;;_' else [term]
    choices = []
    for piece in pieces:
        value = spec.downTerm(piece)
        if value is None:
            raise RuntimeError('Native result not a finite term set: ' + str(piece)[:1000])
        assert matches(a, value) and matches(b, value), (left, right, str(value))
        choices.append(str(value))
    choices = sorted(set(choices))
    assert choices
    # Independent minimality check of the returned native antichain.
    for i, x in enumerate(choices):
        for y in choices[i+1:]:
            assert not matches(spec.parseTerm(x), spec.parseTerm(y))
            assert not matches(spec.parseTerm(y), spec.parseTerm(x))
    return choices, dict(seconds=seconds, rewrites=rewrites, complete=True, native_entry=entry,
                         native_result=native, verified_generalization=True,
                         verified_antichain=True,
                         peak_rss_mib=peak_rss_mib())


def native_catalogue(trees, groups, vendor, out, theory='ACU', pair_timeout_seconds=None, timeout_policy='fail'):
    from .native_trees import ExprTree, ordered_distance
    from .io import write
    test, spec, tokens, provenance = configure(trees, vendor, theory)
    from . import cache
    from .pair_cache import pair_operation
    context = cache.key({'theory': theory, 'tokens': tokens, 'provenance': provenance,
                         'adapter_sha256': cache.fingerprint(__file__, Path(__file__).with_name('pair_cache.py'))})
    assert timeout_policy in ('fail', 'skip_merge')
    reverse = {v: k for k, v in tokens.items()}
    def encode(t):
        return tokens[t.name] + ('(' + ','.join(encode(c) for c in t.children) + ')' if t.children else '')
    def size(t):
        return 1 + sum(size(c) for c in t.children)
    def pattern_tree(t):
        if t.isVariable():
            return ExprTree('OPEN_PATTERN_VARIABLE', [])
        children = [pattern_tree(c) for c in t.arguments()]
        name = reverse[str(t.symbol())]
        if name in ('sum', 'prod'):
            children.sort(key=lambda c: c.signature())
        return ExprTree(name, children)
    patterns, reps, trace, skipped = [], [], [], []
    start = time.monotonic()
    write(out / 'provenance.json', provenance)
    for gi, group in enumerate(groups):
        ordered = sorted(group, key=lambda i: (size(trees[i]), trees[i].signature(), i))
        pattern = encode(trees[ordered[0]])
        steps = []
        retained = [ordered[0]]
        for step, i in enumerate(ordered[1:], 1):
            right = encode(trees[i])
            before = dict(cluster=gi, clusters_done=len(patterns), clusters_total=len(groups),
                          step=step, members=len(group), source=i, left=pattern, right=right)
            write(out / 'native_progress.json', dict(**before, event='pair_started', elapsed=time.monotonic()-start))
            a, b = spec.parseTerm(pattern), spec.parseTerm(right)
            if pair_timeout_seconds is not None:
                outcome = pair_operation(test, spec, pattern, right, context, pair_timeout_seconds)
                if outcome['status'] == 'timeout' and timeout_policy == 'skip_merge':
                    skipped.append(dict(cluster=gi, step=step, source=i, left=pattern, right=right, **outcome))
                    steps.append(dict(source=i, chosen=pattern, action='retain_previous_pattern_skip_timed_out_merge', **outcome))
                    write(out/'skipped_merges.json', skipped)
                    write(out/'native_progress.json', dict(**before, event='pair_skipped', elapsed=time.monotonic()-start, **outcome))
                    continue
                if outcome['status'] != 'succeeded':
                    raise RuntimeError('Native pair did not complete: '+json.dumps(outcome))
                choices, info = outcome['choices'], {**outcome['info'], 'cache_hit': outcome['cache_hit']}
            elif matches(b, a):
                choices, info = [str(a)], dict(shortcut='left_subsumes_right', complete=True)
            elif matches(a, b):
                choices, info = [str(b)], dict(shortcut='right_subsumes_left', complete=True)
            else:
                choices, info = generalize(test, spec, pattern, right)
            pattern = choices[0]
            retained.append(i)
            steps.append(dict(source=i, all_generalizers=choices, chosen=pattern, **info))
            write(out / 'native_progress.json', dict(**before, event='pair_completed', elapsed=time.monotonic()-start, **info))
        pat = spec.parseTerm(pattern)
        assert all(matches(spec.parseTerm(encode(trees[i])), pat) for i in retained)
        # An ignored member remains in the dataset, but is not falsely claimed
        # to be an instance of this pattern. Later merges may cover it anyway.
        covered = [i for i in group if matches(spec.parseTerm(encode(trees[i])), pat)]
        rep = min(covered, key=lambda i: (ordered_distance(pattern_tree(pat), trees[i]), trees[i].signature(), i))
        patterns.append(pattern)
        reps.append(rep)
        trace.append(dict(members=group, retained_members=retained, pattern_covered_members=covered,
                          pattern_uncovered_members=[i for i in group if i not in covered],
                          steps=steps, finite_source_instance=rep))
        write(out / 'native_partial.json', dict(patterns=patterns, trace=trace))
    write(out / 'native.json', dict(theory=theory, source_pair_order='size', patterns=patterns,
          trace=trace, token_to_original_label=reverse, provenance=provenance,
          pair_reduction_rule='complete native ACUOS2 set; lexical-first normalized generalizer',
                      native_entry='standard; native constraint overload for distinct atoms', exact_subsumption_shortcuts=True,
          bounded_enumeration=False, nary_complete_set_claimed=False,
          pair_timeout_seconds=pair_timeout_seconds, timeout_policy=timeout_policy,
          skipped_merges=skipped, exact_all_member_generalization=not skipped))
    return reps, dict(theory=theory, adapter=('budgeted ACUOS2 + skipped timed-out merges + verified finite source instance'
                      if timeout_policy=='skip_merge' else 'ACUOS2 native APIs + verified finite source instance'),
                      provenance=provenance, pair_timeout_seconds=pair_timeout_seconds,
                      skipped_merges=len(skipped), exact_all_member_generalization=not skipped,
                      unrepresented_native_cluster_members=sum(len(x['pattern_uncovered_members']) for x in trace))

