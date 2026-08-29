Video: https://youtu.be/jNx_F3zbFNE

Set up

set up completed and two tables created without any issue.


1. Scenario A — Lightweight compute: A t3.micro EC2 instance (1 vCPU, 1 GB RAM)

1 instances x 0.0104 USD On Demand hourly cost x 160 hours in a month = 1.664000 USD
On-Demand instances (monthly): 1.664000 USD

Total 12 months cost
19.92 USD
Includes upfront cost

2. Scenario B — Heavy analytics workload: A p3.2xlarge EC2 instance 


Upfront cost
0.00 USD
Monthly cost
2,580.38 USD
Total 12 months cost
30,964.56 USD


Scenario A costs about $1.66 per month. Scenario B costs about $2,580.38 per month for EC2, plus the RDS and S3 costs. I was surprised that the GPU instance costs so much.

While using the calculator, I learned that the cost changes based on the instance type and how many hours it runs.

Scenario B is much more expensive because it uses a powerful GPU and runs 24/7. A GPU is useful for heavy ML work, but it is not worth the cost for simple projects.