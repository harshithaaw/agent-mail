# Person-aware retrieval evaluation

Person-aware retrieval uses the incoming sender and thread to favor relevant sent replies involving that person. It supplements semantic retrieval; it does not replace it.

## DEV and TEST

The checked-in evaluation script defines separate DEV and TEST splits, but no saved person-aware DEV/TEST output is present. The master context file contains an older, non-person-aware 8-query threshold check: 6/8 queries had sufficient examples (75%), with 11 usable examples total. These figures are legacy context, not person-aware DEV or TEST results.

## Threshold sweep

The sweep below is switch-on on 12 DEV queries. Gold kept was 0.75 at every limit. Zero-example share is shown per limit.

| Limit | Gold kept | Zero-example share |
|---:|---:|---:|
| 0.50 | 0.75 | 0.17 |
| 0.55 | 0.75 | 0.17 |
| 0.60 | 0.75 | 0.08 |
| 0.65 | 0.75 | 0.08 |
| 0.70 | 0.75 | 0.00 |
| 0.75 | 0.75 | 0.00 |
| 0.80 | 0.75 | 0.00 |

Switch-off gold kept was 0.50 at every limit. With only 12 DEV queries, this is a small sensitivity check. The 0.60 limit was fixed before the sweep; the sweep does not select or validate a new limit. TEST results should be recorded separately when available.

## Live test and limits

In the live test, 3 switch-on drafts retrieved 3 examples each. All examples were dropped by the 0.60 rule, so none appeared in the prompt.

The old path uses a 0.65 limit. The person-aware path uses 0.60.

## Next step

Try a K-Means persona fallback when person-specific retrieval has too little useful context.
