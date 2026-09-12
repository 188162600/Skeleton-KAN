"""Source-only native-count shortfall rule; no held-out scores enter it."""

def select_representatives(results, target):
    good = [r for r in results if r['status'] == 'succeeded']
    exact = [r for r in good if r['final_distinct_count'] == target]
    if exact:
        chosen = min(exact, key=lambda r: r['native_k'])
        return chosen, chosen['representatives'][:], [], True
    lows = [r for r in good if r['final_distinct_count'] < target]
    fillable = []
    for lower in lows:
        supply = {x['signature'] for x in lower['representatives']}
        for higher in good:
            if higher['native_k'] > lower['native_k']:
                supply.update(x['signature'] for x in higher['representatives'])
        if len(supply) >= target:
            fillable.append(lower)
    candidates = fillable or lows
    chosen = min(candidates, key=lambda r: (target-r['final_distinct_count'], r['native_k'])) if candidates else None
    selected = chosen['representatives'][:] if chosen else []
    added = []
    if chosen:
        used = {x['signature'] for x in selected}
        for result in sorted(good, key=lambda r: r['native_k']):
            if result['native_k'] <= chosen['native_k']:
                continue
            for item in result['representatives']:
                if len(selected) == target:
                    break
                if item['signature'] not in used:
                    used.add(item['signature'])
                    selected.append(item)
                    added.append({'from_native_k': result['native_k'], **item})
    return chosen, selected, added, False


def next_count(results, target, upper=225):
    if not 1 <= target <= upper:
        raise ValueError('Invalid target cardinality')
    if not results:
        return target
    if len(select_representatives(results, target)[1]) == target:
        return None
    latest = max(results, key=lambda r: r['native_k'])
    if latest['status'] != 'succeeded':
        return None
    shortage = target-latest['final_distinct_count']
    if shortage <= 0:
        return None
    following = min(upper, latest['native_k']+shortage)
    seen = {r['native_k'] for r in results}
    return following if following > latest['native_k'] and following not in seen else None

