import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
old_run_id = "2192c7b438954528a98cacf76b357e76"

old = client.get_run(old_run_id)
exp_id = old.info.experiment_id
params = old.data.params
tags = old.data.tags

metrics_ok = {}
for name in old.data.metrics.keys():
    hist = client.get_metric_history(old_run_id, name)
    filtered = [m for m in hist if m.step is None or m.step < 100_000]
    if filtered:
        metrics_ok[name] = filtered

with mlflow.start_run(experiment_id=exp_id) as new:
    mlflow.log_params(params)
    mlflow.set_tags(tags)
    for name, values in metrics_ok.items():
        for v in values:
            mlflow.log_metric(name, v.value, step=v.step, timestamp=v.timestamp)

client.delete_run(old_run_id)
print(f"Старый run {old_run_id} удалён, новый run {new.info.run_id} создан.")
