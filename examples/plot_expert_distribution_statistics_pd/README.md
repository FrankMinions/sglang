# How to get expert distribution statistics

For prefill and decode nodes, you need add the following parameters:
```bash
--expert-distribution-recorder-mode stat
--disable-overlap-schedule
--expert-distribution-recorder-buffer-size -1
```
and use `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` to dump .pt files.

Then you can start statistics based on the master of prefill and decode nodes：
```bash
curl -X POST "http://localhost:8001/start_expert_distribution_record"
```

After you have run a large amount of data, you need to stop the statistics and save the data to disk:
```bash
curl -X POST "http://localhost:8001/stop_expert_distribution_record"
curl -X POST "http://localhost:8001/dump_expert_distribution_record"
```
