import json

d = json.load(open('scripts/precomputed_results.json'))

print("Female FALSE_NEGATIVE cases with DPR at decision time:")
fns = [x for x in d['all_results'] if x['gender'] == 'Female' and x['result_type'] == 'FALSE_NEGATIVE']

for x in fns:
    print(f"  Row {x['row_index']}: prob={x['base_probability']:.3f}, DPR={x['current_dpr_at_decision']:.3f}, intervention={x['intervention_occurred']}")

print(f"\nTotal female false negatives: {len(fns)}")
print(f"With intervention: {sum(1 for x in fns if x['intervention_occurred'])}")
