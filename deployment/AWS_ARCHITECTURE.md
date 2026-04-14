# PulseRAG — AWS Deployment Architecture

## Overview

PulseRAG is deployed on AWS using a containerized architecture that separates compute, storage, and serving concerns. The system is designed to scale horizontally for concurrent users while keeping ML model artifacts and vector index state persistent across deployments.

---

## Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
                         │              INTERNET                    │
                         └─────────────────┬───────────────────────┘
                                           │
                         ┌─────────────────▼───────────────────────┐
                         │         Amazon CloudFront (CDN)          │
                         │   Static asset caching · HTTPS offload   │
                         └─────────────────┬───────────────────────┘
                                           │
                         ┌─────────────────▼───────────────────────┐
                         │        Amazon API Gateway (HTTP)         │
                         │   Rate limiting · Auth · CORS headers    │
                         └──────┬──────────────────────┬───────────┘
                                │                      │
               ┌────────────────▼──────┐   ┌──────────▼────────────────┐
               │   AWS Lambda          │   │   Amazon ECS (Fargate)    │
               │   (Lightweight API)   │   │   PulseRAG Streamlit App  │
               │   · Health checks     │   │   Docker container        │
               │   · Webhook triggers  │   │   cpu=1vCPU mem=2GB       │
               │   · Drift alerts      │   │   Auto-scaling: 1–4 tasks │
               └────────────────┬──────┘   └──────────┬────────────────┘
                                │                      │
               ┌────────────────▼──────────────────────▼────────────────┐
               │                  Amazon EFS (Elastic File System)       │
               │   Shared persistent storage mounted across all tasks    │
               │   · chroma_db/     ← ChromaDB HNSW vector index        │
               │   · pulserag.db    ← SQLite interaction log            │
               │   · pulserag_model.pkl ← Trained ML model              │
               │   · mlruns/        ← MLflow experiment artifacts       │
               └───────────────────────────┬────────────────────────────┘
                                           │
               ┌───────────────────────────▼────────────────────────────┐
               │                     Amazon S3                          │
               │   · Model backups (versioned)                          │
               │   · Monitoring reports (monitoring_report.html)        │
               │   · Synthetic data snapshots                           │
               │   · CI/CD build artifacts                              │
               └───────────────────────────┬────────────────────────────┘
                                           │
               ┌───────────────────────────▼────────────────────────────┐
               │              Amazon CloudWatch                         │
               │   · Application logs (ECS task logs)                   │
               │   · Custom metrics: understanding_rate, latency_p95    │
               │   · Alarms: PSI drift > 0.20, error rate > 5%         │
               │   · Dashboard: real-time pipeline health               │
               └────────────────────────────────────────────────────────┘
```

---

## Services Used & Justification

### Amazon ECS (Elastic Container Service) with Fargate
**Why**: Runs the PulseRAG Docker container without managing EC2 instances. Fargate is serverless compute for containers — you pay per task CPU/memory second. Auto-scaling based on CPU utilization handles variable user load.

**Config**:
- Task definition: 1 vCPU, 2GB RAM (sufficient for all-MiniLM-L6-v2 + ChromaDB)
- Service: min 1 task, max 4 tasks, scale-out at 70% CPU
- Container port: 8501 (Streamlit)

### Amazon EFS (Elastic File System)
**Why**: ChromaDB and SQLite require persistent disk that survives container restarts and is shared across all ECS task replicas. EFS provides a POSIX-compatible NFS mount that all Fargate tasks can read/write simultaneously.

**Mount**: `/app/chroma_db`, `/app/data` → EFS volume

### Amazon S3
**Why**: Object storage for model artifacts, monitoring reports, and backups. S3 versioning enables model rollback. Lifecycle policies move old artifacts to Glacier after 90 days.

**Buckets**:
- `pulserag-models/` — versioned `.pkl` files with metadata tags
- `pulserag-reports/` — Evidently HTML reports, one per drift check
- `pulserag-data/` — training data snapshots for audit trail

### Amazon API Gateway
**Why**: Sits in front of ECS to provide rate limiting (100 req/min per IP), API key auth for programmatic access, and request/response logging. Also routes lightweight webhook calls to Lambda without spinning up a full container.

### AWS Lambda
**Why**: Handles event-driven tasks that don't need a persistent container. Specifically used for: drift alert webhooks (triggered by CloudWatch alarm when PSI > 0.20), scheduled health pings, and S3 backup triggers on model retrain.

### Amazon CloudWatch
**Why**: Centralized logging and monitoring. ECS task logs are automatically shipped. Custom metrics pushed from the app include `understanding_rate`, `avg_retrieval_score`, `latency_p95`, and `hallucination_risk`. Alarms trigger SNS notifications and Lambda auto-remediation.

### Amazon CloudFront
**Why**: CDN layer that caches static Streamlit assets (JS, CSS) at edge locations globally. Reduces latency for international users and offloads HTTPS termination from the application tier.

---

## Deployment Pipeline (CI/CD → AWS)

```
GitHub Push (main)
       │
       ▼
GitHub Actions CI
  ├── Lint (flake8, black)
  ├── Unit tests (pytest 35 tests)
  ├── Docker build + validate
  └── MLflow experiment log
       │
       ▼ (on success)
Amazon ECR (Elastic Container Registry)
  └── Push image: pulserag:${git_sha}
       │
       ▼
Amazon ECS
  └── Update service → rolling deploy
      (new task starts → health check passes → old task stops)
       │
       ▼
CloudWatch
  └── Monitor new deployment for 5 min
      └── Auto-rollback if error rate > 5%
```

### Extending CI/CD for AWS (add to ci.yml):
```yaml
  deploy:
    needs: [test, docker]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Push to ECR
        run: |
          docker tag pulserag:${{ github.sha }} \
            ${{ secrets.ECR_REGISTRY }}/pulserag:${{ github.sha }}
          docker push ${{ secrets.ECR_REGISTRY }}/pulserag:${{ github.sha }}

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster pulserag-cluster \
            --service pulserag-service \
            --force-new-deployment
```

---

## Environment Variables (AWS Secrets Manager)

| Variable | Description | Source |
|---|---|---|
| `GROQ_API_KEY` | Groq LLM API key | AWS Secrets Manager |
| `PULSERAG_DEVICE` | `cpu` or `cuda` | ECS task definition env |
| `DB_PATH` | `/app/data/pulserag.db` | ECS task definition env |
| `CHROMA_DIR` | `/app/chroma_db` | ECS task definition env |
| `MLFLOW_TRACKING_URI` | S3 or EC2 MLflow server URI | AWS Secrets Manager |

---

## Cost Estimate (Free Tier / Low Traffic)

| Service | Config | Est. Monthly Cost |
|---|---|---|
| ECS Fargate | 1 task × 730 hrs × 0.25 vCPU | ~$10 |
| EFS | 5GB storage | ~$1.50 |
| S3 | 10GB storage | ~$0.23 |
| API Gateway | 1M requests | ~$3.50 |
| CloudWatch | Basic metrics + logs | ~$2 |
| Lambda | 100K invocations | Free tier |
| **Total** | | **~$17/month** |

---

## Scaling Strategy

**Horizontal scaling**: ECS service auto-scales from 1 to 4 Fargate tasks based on CPU. ChromaDB on EFS supports concurrent reads from multiple tasks. SQLite on EFS handles concurrent writes via WAL mode (enabled at startup).

**Model serving at scale**: For >100 concurrent users, the ML model and ChromaDB collection would be moved to dedicated services — Amazon SageMaker real-time endpoint for inference and a managed vector DB (Pinecone or Weaviate) for retrieval. The application code is already abstracted to make this a backend swap.

**Database at scale**: SQLite → Amazon RDS PostgreSQL. The `_insert()` and `_load_table()` functions in `backend.py` are the only points that need changing — they already use parameterized queries compatible with psycopg2.