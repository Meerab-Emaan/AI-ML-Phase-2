# Task 5: Auto Tagging Support Tickets Using LLM

**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## Objective

Automatically classify and tag customer support tickets into predefined categories using a Large Language Model (LLM), comparing zero-shot and few-shot learning approaches.

---

## Dataset

- **Type:** Free-text Support Ticket Dataset
- **Source:** Custom dataset created for this task
- **Size:** 15 labeled support tickets
- **Categories (Tags):** Network, Account, Bug, Billing, How-To

---

## Methodology / Approach

### Step 1 — Dataset Creation
- Created a dataset of 15 realistic support tickets
- Each ticket is labeled with one of 5 tags: Network, Account, Bug, Billing, How-To
- Tickets cover a wide range of real-world customer issues

### Step 2 — Zero-Shot Classification
- Used `facebook/bart-large-mnli` via Hugging Face `zero-shot-classification` pipeline
- No training examples provided — the model classifies based on label names alone
- Returned **Top 3 most probable tags** with confidence scores for each ticket

### Step 3 — Few-Shot Learning
- Enhanced the prompt with 5 labeled examples (one per category)
- The same model was re-run with the enriched prompt
- Compared accuracy improvement over zero-shot baseline

### Step 4 — Comparison & Evaluation
- Compared **Zero-Shot vs Few-Shot accuracy** on all 15 tickets
- Plotted accuracy comparison bar chart
- Plotted tag distribution across the dataset

### Step 5 — Deployment
- Deployed using **Gradio** interface
- Users can enter any support ticket and get:
  - Top 3 predicted tags with confidence scores
  - Both zero-shot and few-shot results side by side

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Hugging Face Transformers | Zero-shot classification pipeline |
| facebook/bart-large-mnli | Pre-trained NLI model for classification |
| Pandas / NumPy | Data processing |
| Matplotlib | Result visualization |
| Gradio | Live demo deployment |

---

## Key Results & Observations

| Method | Accuracy |
|--------|----------|
| Zero-Shot | ~73% |
| Few-Shot | ~80% |
| Improvement | ~7% |

**Sample Prediction:**
```
Ticket: "My internet keeps disconnecting every hour"

Zero-Shot Top 3:    Few-Shot Top 3:
1. Network  82%     1. Network  89%
2. Bug      10%     2. Bug       7%
3. Account   5%     3. Account   3%
```

- Few-shot prompting improved accuracy by ~7% with just 5 examples
- **Network** and **Bug** categories were sometimes confused (both relate to technical issues)
- **Billing** tickets had the highest accuracy in both approaches
- Zero-shot classification works well out-of-the-box without any training data
- Few-shot examples act as a simple and effective way to guide the model

---

## Skills Gained

- Prompt engineering
- LLM-based text classification
- Zero-shot and few-shot learning techniques
- Multi-class prediction and ranking (Top 3 tags)

---

## Repository Structure

```
task5_ticket_tagging/
├── task5_ticket_tagging.ipynb    # Main notebook
├── accuracy_comparison.png       # Zero-shot vs Few-shot chart
├── tag_distribution.png          # Ticket tag distribution chart
├── README.md                     # This file
```

---

## How to Run

```bash
# Install dependencies
pip install transformers datasets scikit-learn pandas numpy gradio

# Open the notebook
jupyter notebook task5_ticket_tagging.ipynb
```


