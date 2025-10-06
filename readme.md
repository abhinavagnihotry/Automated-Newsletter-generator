# 📩 Automated AI Newsletter Generator

## Context
Companies are actively exploring and applying **AI/LLM technologies** across our products and internal workflows.  
To keep up with this fast-moving field, we want to create a **weekly AI newsletter** for the employees.  

A newsletter helps us:  
- **Stay ahead of industry trends** by surfacing the latest AI developments  
- **Discover new use cases** that could inspire or accelerate our own projects  
- **Reduce information overload** by filtering and summarizing only the most relevant updates  
- **Share knowledge internally** in a concise, digestible, and engaging format  

The task was to design and implement a pipeline that automatically generates this newsletter.

We use the dataset of **AI/LLM-related news, research, and use cases** curated by the **ZenML team** and published on **Hugging Face Datasets**: 👉 [zenml/llmops-database](https://huggingface.co/datasets/zenml/llmops-database)

---

## Plan
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

## Output
Final output newsletter: *newsletter.md*

