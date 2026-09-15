# Project 08 — Cloud Setup and Cost Analysis

## Video

https://youtu.be/jNx_F3zbFNE

## Part A — Supabase Setup

I completed my Supabase setup and created the `weather_raw` and `weather_enriched` tables without any issues.

## Part B — Cloud Cost Analysis

Scenario A uses a `t3.micro` EC2 instance for 160 hours per month. It costs about **$1.66 per month** and **$19.92 for 12 months**.

Scenario B uses a `p3.2xlarge` GPU EC2 instance, an RDS `db.m5.large` database, and 1 TB of S3 storage. The EC2 costs **$2,233.80**, RDS costs **$323.03**, and S3 costs **$23.55** per month. The total is about **$2,580.38 per month** or **$30,964.56 for 12 months**.

I was surprised by how expensive the GPU instance is. I also learned that the cost changes a lot depending on the type of instance and how many hours it runs. Scenario B is much more expensive because it uses a powerful GPU and runs 24/7. A GPU can be useful for heavy machine learning work, but for a small project, it would probably not be worth the cost.
