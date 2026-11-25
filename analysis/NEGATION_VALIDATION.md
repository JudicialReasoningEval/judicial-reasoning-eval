# M3 polarity-pair validation protocol

The probe exports `results/_negation_probe_validation_sample.csv`. Its labels
must be completed by people; neither the rule system nor an LLM may fill the
blank columns.

Annotate each row independently, without looking at its cosine score:

- `grammatical_original`, `grammatical_perturbed`: `yes` or `no`.
- `relation`: `contradiction`, `scope_change`, `neutral`, `malformed`, or
  `uncertain`. Use `contradiction` only when the edited proposition cannot be
  true in the same interpretation as the original. Use `scope_change` for a
  meaningful polarity change inside an attribution, modality, condition,
  disjunction, or subsidiary clause that does not contradict the sentence's
  main assertion.
- `negation_scope`: `main_assertion`, `embedded_claim`, `modality`,
  `condition`, `disjunction`, `citation_metadata`, or `other`.
- `legal_materiality`: `core`, `supporting`, or `peripheral`, judged relative
  to the legal reasoning expressed by the sentence.
- `notes`: a short explanation for `malformed` or `uncertain`.

For a paper-quality estimate:

1. Have two Italian-speaking annotators label the rows independently.
2. Report acceptance precision for grammatical `contradiction` pairs, with a
   binomial confidence interval, and Cohen's kappa for `relation`.
3. Adjudicate disagreements before using pairs as a gold diagnostic.
4. Report `scope_change` separately; do not silently merge it with
   `contradiction`.
5. Add a separate, human-authored set for the Article 318/319 distinction.
   Keep it separate from the representative corpus sample so that targeted
   oversampling does not distort the detector-precision estimate.

Until those fields are completed, the JSON deliberately calls the automatic
rows *candidate polarity changes*, never contradictions.

## Real M3-selected pairs

`python -m analysis.m3_polarity_audit` creates
`results/_m3_polarity_audit_sample.csv`. These are real judge/model argmax
pairs where exactly one side contains an explicit `non`, prioritised at cosine
at least 0.80 for annotation economy. The 0.80 cutoff is not an M3 threshold
and must not be described as one.

For this sheet, label `relation` as `contradiction`, `compatible`,
`different_scope`, `partial_overlap`, or `uncertain`. A cue mismatch alone is
not an error: one sentence may simply include an additional subordinate clause
or citation parenthetical. Only adjudicated `contradiction` rows may enter an
estimate of realised M3 inflation.
