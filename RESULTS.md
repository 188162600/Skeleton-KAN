# Complete benchmark results

Download [skeleton-kan-results-v1.0.0.zip](https://github.com/188162600/Skeleton-KAN/releases/download/v1.0.0/skeleton-kan-results-v1.0.0.zip)
from the public release, or run:

```sh
python download_results.py
```

Archive size: **56,212,586 bytes**. SHA-256:

```text
9e26ab0212f3e09e92fede0fe312020669d2ec8a4988e3f0d7155b3d5d22b46e
```

Extract to a directory of your choice, then run the enclosed `verify_results.py`
from the extracted result root. The archive contains **12,960 complete
configurations and 129,600 seed records**, covering all eight methods on
Transfer-80 and PDE-10. It retains every configuration, not just winners:

```text
model-name/task-name/configuration-name/
  result.json
  training_config.json
  provenance.json
```

The results archive is byte-identical to the audited review export. Its existing
privacy/anonymization notes describe removal of machine details, not a requirement
to anonymize this public repository. Numerical values, parameter counts, seed
records and configuration linkage are unchanged. Checkpoints, full training logs
and sampled arrays are not included.

For each equation, select **one** configuration by mean log validation NMSE across
all ten seeds after all 18 candidates finish. Test scores are report-only.
Main GMSE/GNMSE metrics have no error floor. Never choose a different winning
builder independently for each seed.

These records describe the completed experiments. They do not claim that a run
on another operating system, GPU, library version or compilation backend will
be bitwise identical. Use the repository's frozen launch configurations and
document any environmental or configuration changes.
