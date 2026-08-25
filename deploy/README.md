# Deploying Distill

The app splits cleanly: only `/api/extract` needs a model, everything else is
ordinary Python. So the deployment cost is decided entirely by which backend
you serve.

| Backend | Params | Hardware | Latency | Rough AWS cost, always on |
| --- | --- | --- | --- | --- |
| `vlm` — Qwen2.5-VL-3B | 3B | GPU, ~8GB VRAM | 4.9s p50 (k=3) | ~$380/mo (`g4dn.xlarge`) |
| `donut` — fine-tuned | 200M | **CPU** | ~5.0s, measured | ~$0–15/mo (App Runner, Fargate) |

**The win is the hardware bill, not the clock.** Donut runs at 0.16s/doc on a
GPU but roughly **5.0s/doc on CPU** — about the same wall time as the VLM on a
GPU. What changes is that it needs no GPU at all, which is the difference
between a ~$380/month instance and a ~$15/month container.

The Dockerfile builds the CPU path. Set `DISTILL_BACKEND=vlm` only where a GPU
is actually present.

## Build

The image expects the model at `model/`, so copy in the checkpoint you want to
serve. Use `-best`, the best-validation checkpoint, not the last epoch: later
epochs can overfit, and the final one is not automatically the one to ship.

```bash
cp -r finetune/donut-distill-best model
docker build -t distill:latest .
docker run --rm -p 8080:8080 distill:latest
curl localhost:8080/healthz
```

`/healthz` answers as soon as the process is up and reports `model_ready`
separately, so an orchestrator will not kill the container during the model
load — **measured at ~52s on CPU**, which is why the healthcheck allows a 90s
start period.

## AWS

**App Runner** is the shortest path: it takes a container from ECR and gives
you HTTPS and a URL. Push the image, point a service at it, set the port to
8080 and the health check to `/healthz`.

**Fargate** if you want it inside a VPC or need more control. Same image, with
an ALB in front targeting `/healthz`.

**Lambda** is cheapest at demo traffic and the free tier likely covers it
outright, but it needs the Lambda Web Adapter or Mangum in front of ASGI, and a
cold start pays the full ~52s model load before a ~5s extraction. That is a bad
first impression for a portfolio link. Prefer App Runner, or a small
always-warm Fargate task, unless traffic is genuinely negligible.

### Things that will bite

- **Uploads land in `/app/outputs`**, which is container-local and vanishes on
  redeploy. Fine for a demo; mount EFS or write to S3 if corrections need to
  survive.
- **The eval queue needs the dataset**, which is deliberately not in the image.
  Without it the app degrades to upload-only rather than failing to boot.
- **`HF_HUB_OFFLINE=1`** is set so a network hiccup cannot become a silent
  download attempt at request time. Remove it only if you intend to fetch
  weights at runtime.
