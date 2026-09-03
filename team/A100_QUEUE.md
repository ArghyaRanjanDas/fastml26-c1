# A100 job queue (Purdue AF, agent hh4b runs it)

The A100 (40 GB) is fully ours until the hackathon ends (Friday). Any agent can append a job here;
hh4b pulls this file every ~10 minutes, runs jobs top to bottom, and writes the result under the job.

Rules: one job = one fenced command block that runs from the repo root on the AF (`/work/users/das214/fastml26/fastml26-c1`,
venv `../venv`, caches in `team/cache/{train1M,train300k,eval100k}` with raw X 16x5, F 11, y, group; rich
channels are computed on the fly by `team/data.py`). Put the expected runtime and the number to beat.
hh4b: mark a job `[running]`, then `[done: <AUC etc>]` and commit+push; never delete jobs.

## Jobs

