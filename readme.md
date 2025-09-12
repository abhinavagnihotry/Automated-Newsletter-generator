# 📩 Assignment: Automated AI Newsletter Generator

## Context
Odido is actively exploring and applying **AI/LLM technologies** across our products and internal workflows.  
To keep up with this fast-moving field, we want to create a **weekly AI newsletter** for Odido employees.  

A newsletter helps us:  
- **Stay ahead of industry trends** by surfacing the latest AI developments  
- **Discover new use cases** that could inspire or accelerate our own projects  
- **Reduce information overload** by filtering and summarizing only the most relevant updates  
- **Share knowledge internally** in a concise, digestible, and engaging format  

Your task is to design and implement a pipeline that automatically generates this newsletter.

To simplify the task for this assignment, we provide a dataset of **AI/LLM-related news, research, and use cases** curated by the **ZenML team** and published on **Hugging Face Datasets**: 👉 [zenml/llmops-database](https://huggingface.co/datasets/zenml/llmops-database)

---

## Your Task
Build an **end-to-end pipeline** that:

1. **Ingests the dataset**  
   - Use the Hugging Face dataset.  
   - Each record contains metadata such as `created_at`, `title`, `industry`, `source_url`, `company`, etc. See the dataset page on Hugging Face for full details.

2. **Processes & Analyzes**  
   - Identify the most relevant or trending items for the current week.  
   - Categorize items into meaningful sections (e.g., *Research Highlights*, *Industry News*, *Cool Use Cases*).  
   - Summarize use cases into clear, concise, newsletter-friendly text.  

3. **Generates a Newsletter**  
   - Produce the newsletter in **Markdown format**.  
   - Include:  
     - A short introduction  
     - Categorized sections with summaries and links  
     - A short closing section  

4. **Automates the Workflow**  
   - Ensure the pipeline can be run on a weekly schedule.  
   - Handle updates gracefully (avoid duplicates; include only fresh items).  

---

## Requirements
- Use an **LLM** (e.g., OpenAI API or an open-source model) for summarization/writing.  
- Write **modular, clean, and well-documented code**.  
- Provide a **sample newsletter** generated from the Hugging Face dataset.  
- Provide clear instructions on how to run your solution locally.  

**Bonus points if you:**  
- Deploy the pipeline with automation (e.g., GitHub Actions or another scheduler).  
- Add personalization (e.g., selecting sections based on reader profile: technical or non-technical).  
- Evaluate or improve summary quality.  

---

## Deliverables
Please submit a **GitHub repository** containing:  
- 📂 Source code for the pipeline  
- 📝 `README.md` with setup instructions and design explanation  
- 📄 A sample generated newsletter (`newsletter.md`)  
- ⚙️ Any configuration files (e.g., `.env.example`)  

---

## Evaluation Criteria
We’ll evaluate based on:  
- **Data Handling** – How well you ingest, filter, and process the Hugging Face dataset  
- **LLM Usage** – Effective prompt design and output quality  
- **Code Quality** – Structure, clarity, documentation, reproducibility  
- **Automation** – Ability to set up a repeatable, scheduled workflow  
- **Creativity** – Newsletter structure, style, polish  
- **Bonus** – Deployment, personalization, or evaluation methods  

---

## Time Expectation
We estimate this assignment will take **3–4 hours**. Please focus on **building a working prototype**, not on perfection.  

---

✅ **What you’ll get from us**  
- The Hugging Face dataset link 👉 [zenml/llmops-database](https://huggingface.co/datasets/zenml/llmops-database).  

✅ **What we expect from you**  
- A working pipeline that generates a weekly newsletter.  
- Sample newsletter outputs in **Markdown**.  
- Your code and documentation.  
