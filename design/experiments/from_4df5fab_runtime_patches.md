Base commit: 4df5fab

Learned components source: 4df5fab stable assets

Runtime patches source: newer runtime-patch branch / current newer code, but only runtime logic is eligible for merge

Experiment goal: preserve malignant recall while trying to recover top1/topk stability

Constraints:
- Do not modify `master` directly.
- Do not replace stable learned controller/retriever checkpoints with mini-retrain artifacts.
- Only merge runtime patches in controlled steps.
