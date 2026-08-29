
Part 1: Warmup — Cloud Concepts


Cloud Concepts Question 1


 What is the core economic model of cloud computing, and how does it differ from owning your own servers? 

 The core economic model of cloud computing is pay as you go. You only pay for what you use. If you need more compute or more resources you only need to do a simple click and you can scale vertically or horizontally as you need.

 Cloud Concepts Question2

  What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.
  
   Vertical scaling when you scale on your current virtual machine(Ec2). You can add more memory, more storage, cpu… Horizontal scaling when you deploy and add more machines responding to a current or a future increase and high demand on your website or app. 
   
   Then, for the three scenarios below, write one sentence saying which type of scaling applies and why. 
   
   A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch: Horizontal scaling will be better because there is huge increase in the number of users.Deploying many machines will be a good choice. 
   
   A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM. In this case a vertical scaling will be enough, because the data scientist only need more gpu and ram and can have that by scaling vertically.
   
    A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machine Horizontal scaling will be the best choice here as long as the work can be split across machines so adding more machines will be a good option.

Cloud Concepts Question 3 

Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each.
 
SaaS: Gmail GitHub Codespaces
Iaas: Azure Virtual Machines AWS S3 (Simple Storage Service) 
PaaS: Snowflake Azure App Service

IaaS: is a kind of cloud service where the cloud provider provision infrastructures as a resource for the user. As a developer or user I will be responsible for deploying my application, and responsible for the security of my application . An example is AWS Ec2 which you chose the type of operating system, the size of RAM, cpu and storage 

PaaS: is a cloud service when the cloud provider provide all necessary infrastructures and platform to run your application or software. The provider manage the infrastructure, but you bring your own code. An example is Azure app service which is a platform for hosting web applications. 

SaaS: Software as a service is cloud service where the cloud provider provides every thing for you you only manage your security and your data. Everything else managed by the provider. An example is facebook, Turbo Tax service 

Cloud Concepts Question 4 

What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up? 

managed data platform like Databricks or Snowflake take a different approach: they pre-wire the pieces for you, optimizing specifically for data and analytics workloads. provisions and manages cloud resources on your behalf. This makes it much faster to get started with large-scale data processing or machine learning, at the cost of some flexibility and, potentially higher costs. 

Cloud Concepts Question 5 

The lesson names two situations where the cloud is probably not the right choice. What are they? The cloud isn't the right tool for every problem. The cloud is likely not the right choice if your dataset fits on a single machine or if you lack massive compute demands, making local processing faster and cheaper for initial prototypes. Additionally, the steep learning curve and potential for high costs make it unsuitable when quick, simple, and inexpensive solutions are needed for smaller tasks.


Part 2: Warmup — Cloud Landscape


Cloud Landscape Question 1
AWS: AWS has many cloud services and is good for large companies and startups.
GCP: GCP is strong in data and machine learning and is good for companies working with data and AI.
Azure: Azure works well with Microsoft tools and is common in large companies and government organizations.

Cloud Landscape Question 2
Access: Supabase is easier to create and students can start quickly.
Learning: Supabase uses a SQL database with rows and columns, which is useful for learning data skills.
Pipeline: Supabase makes the ETL pipeline easier to understand because the raw and enriched data can be stored in two tables.

Reflection: I should choose a cloud tool that is easy to use, fits my project, and helps me learn useful skills.

Cloud Landscape Question 3
Object storage (AWS S3): Store the 10 TB of image files.
Compute (AWS EC2): Run the ML training on a GPU and stop it when finished.
Serverless compute ( Google Cloud Functions): Run the web API and scale when needed.
LLM API (Azure OpenAI): Send data to an LLM and get a text response.

Cloud Landscape Question 4

I can build a weather data project. I can store weather data in Supabase and use Azure OpenAI to analyze the data.

Using one cloud provider can make the project easier to manage. But I may lose some useful tools that are better on other providers.