# Part 1: Warmup — Cloud Concepts

## Cloud Concepts Question 1

The main economic model of cloud computing is **pay as you go**. You pay only for the resources you use instead of buying and maintaining your own physical servers.

## Cloud Concepts Question 2

**Vertical scaling** means making one machine stronger by adding more CPU, RAM, or storage. For example, I can upgrade an EC2 instance if my application needs more memory.

**Horizontal scaling** means adding more machines to share the work. For example, if a website gets a lot more users, I can add more servers.

* **Web app with 100,000 users:** Horizontal scaling is better because more machines can help handle the large increase in users.
* **ML training needs a faster GPU and more RAM:** Vertical scaling is better because the data scientist needs a more powerful machine.
* **Data pipeline goes from 10 to 10,000 files:** Horizontal scaling is better because the files can be split between many machines.

## Cloud Concepts Question 3

* **Gmail — SaaS:** Gmail is complete software that users access online without managing the servers.
* **Azure Virtual Machines — IaaS:** Azure provides the virtual machine, but the developer manages the operating system and applications.
* **AWS S3 — IaaS:** AWS provides cloud storage where developers can store and manage files and data.
* **GitHub Codespaces — PaaS:** It gives developers a ready-to-use development environment without having to set up the infrastructure.
* **Snowflake — PaaS:** It provides a managed platform for storing, processing, and analyzing data.
* **Supabase — BaaS:** It provides backend services such as a database, authentication, and APIs for applications.

**IaaS:** Infrastructure as a Service gives you resources like virtual machines and storage. The provider manages the physical hardware, but I manage things like the operating system and my application. An example is **Azure Virtual Machines**.

**PaaS:** Platform as a Service gives me a platform where I can build and run applications without managing most of the infrastructure. I mainly manage my code and data. An example is **GitHub Codespaces**.

**SaaS:** Software as a Service gives me a complete application that is ready to use. I mostly manage my account and data. An example is **Gmail**.

## Cloud Concepts Question 4

A managed data platform like Snowflake or Databricks provides tools that are already set up for working with large amounts of data. It is easier and faster than setting up everything myself with AWS or GCP. I gain simplicity and save time, but I give up some control and it may cost more.

## Cloud Concepts Question 5

The cloud may not be the best choice in two situations:

1. If the data and work can easily fit and run on one local computer.
2. If the cloud cost and complexity are more than what the small project needs.

# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1

**AWS:** AWS has many cloud services and is used by many startups and large companies.

**Microsoft Azure:** Azure works well with Microsoft products and is common in large companies and government organizations.

**Google Cloud Platform (GCP):** GCP is strong in data, machine learning, and AI and is useful for companies working with these technologies.

## Cloud Landscape Question 2

**Access:** Supabase is easier to create and students can start using it quickly.

**Learning:** Supabase uses PostgreSQL and SQL tables, which helps students practice useful database and data skills.

**Pipeline:** Supabase makes the data pipeline easier to understand because raw and enriched data can be stored in separate tables.

**Reflection:** This shows me that I should not choose a cloud tool only because it is popular. I should choose one that is easy to use, fits my project, and helps me learn the skills I need.

## Cloud Landscape Question 3

1. **Object storage — AWS S3:** I can use S3 to store 10 TB of image files and access them from different machines.

2. **Compute — AWS EC2:** I can use a GPU EC2 instance to train the ML model for four hours and then shut it down.

3. **Serverless compute — Google Cloud Functions:** I can use it to run a web API and automatically scale when traffic changes.

4. **LLM API — Azure OpenAI:** I can send structured data to an LLM and receive a text response.

## Cloud Landscape Question 4

I can build a weather data project using **Supabase** to store weather data and **Azure OpenAI** to analyze the data and create a short summary.

Using one provider can make the project easier to manage because everything is in one place. However, I may give up useful tools or better services that are available from other providers.
