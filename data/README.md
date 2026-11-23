# Exact release datasets

Dataset bytes live in the private [Lbuk/TheSystem-Data](https://huggingface.co/datasets/Lbuk/TheSystem-Data)
repository. The release manifest pins its revision and every payload hash.
This GitHub folder holds compact manifests, not duplicate dataset downloads.

| Dataset | Train | Validation | Test / checks |
|---|---:|---:|---:|
| Shared answer selection | 18,000 | 2,000 | 2,000 |
| Original `<FAST>` routing | 32,000 | 2,400 | 2,400 |
| Diverse `<FAST>` continuation | 32,000 | 4,000 new + 2,400 original | 4,000 new + 2,400 original |
| Additional answer formats | — | — | 300 |
| Five external benchmarks | — | — | 1,648 |
| Generation checks | — | — | 45 unique prompts |

Rows, order, labels, mapped choices and source identifiers are preserved.
The shared answer splits are stored once. The diverse gate mixture contains
16,000 reused training requests and 16,000 new varied requests. Related variants
share source contexts; request counts do not equal independent scenarios.

The external subsets are the pinned publisher samples: VitaminC 599, PAWS 250,
SQuAD2 answerability 299, Aegis2 250 and PubMedQA 250. PubMedQA uses the published
subset of expert-labelled data, not the official leaderboard split. Manifests
retain selection IDs, upstream revisions and prompt-conversion hashes.

Internal tests have been inspected during development. Source grouping separates
train/validation/test contexts but does not establish absence of pretraining or
all semantic overlap. Preserve the supplied splits rather than reshuffling them.

Read [source licences](../licenses/README.md). Terms apply per source; this mixture
is not covered by one blanket project licence.
