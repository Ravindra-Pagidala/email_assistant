# 📧 AI Email Generation Assistant with Prompt Engineering & Model Evaluation

## Project Overview

This project is an **AI-powered Email Generation Assistant** built to generate professional emails based on:

- **Intent** → Why the email is being written  
- **Key Facts** → Important details that must appear in the email  
- **Tone** → Desired communication style (formal, casual, urgent, empathetic)

Example:

Input:
- Intent: Follow up after job interview  
- Key Facts: Interview date, interviewer name, position applied for  
- Tone: Formal  

Output:
- A professionally written follow-up email maintaining the requested tone.

---

# 🏗️ Architectural Design Decisions

This project follows a **modular layered architecture** instead of putting everything in one file.

Why?

Because in real-world production systems:
- Prompt logic changes often  
- Models may change later  
- Evaluation metrics may evolve  
- New scenarios may be added  

Keeping everything modular makes the project:
- Easier to maintain  
- Easier to debug  
- Easier to scale  

---

## Project Structure

```bash
email_assignment/
│
├── config/
│   └── settings.py
│
├── models/
│   ├── gemini_client.py
│   └── groq_client.py
│
├── prompts/
│   └── advanced_prompt.py
│
├── scenarios/
│   └── test_scenarios.json
│
├── evaluation/
│   ├── metrics.py
│   └── evaluator.py
│
├── results/
│   ├── gemini_results.csv
│   └── groq_results.csv
│
├── reports/
│   └── comparative_analysis.md
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 📂 Folder Explanation

---

## config/

Contains:

### `settings.py`

Responsible for:
- Loading API keys from `.env`
- Managing model configurations
- Keeping secrets/configuration centralized

Think of this like:
> `application.properties` in Spring Boot.

---

## models/

Contains model integration logic.

### `gemini_client.py`
Handles all Gemini/Gemma API calls.

### `groq_client.py`
Handles all Groq/LLaMA API calls.

Why separate files?
Because if one provider changes API tomorrow:
- Only that file needs modification.

---

## prompts/

### `advanced_prompt.py`

Contains prompt templates.

This file handles:
- Role prompting
- Few-shot prompting
- Structured prompt formatting

Why separate?
Because prompt engineering is the **core logic** of LLM apps.

---

## scenarios/

### `test_scenarios.json`

Contains:
- 10 manually created evaluation test cases
- Human-written reference emails

Used for:
- Benchmarking model quality

---

## evaluation/

### `metrics.py`

Contains custom scoring logic for:

1. Fact Integration Score  
2. Tone Consistency Score  
3. Actionability Score  

---

### `evaluator.py`

Acts as evaluation orchestrator.

Responsible for:
- Running all test scenarios  
- Calling models  
- Applying metrics  
- Saving results  

---

## results/

Stores generated CSV outputs after evaluation.

Example:

- Gemini Results
- Groq Results

---

## reports/

Contains final written comparison and recommendation report.

---

# 🧠 Prompt Engineering Techniques Used

This system uses advanced prompting strategies:

### 1. Role Prompting
Makes model behave like a professional email writer.

---

### 2. Few-Shot Prompting
Provides examples of ideal outputs.

---

### 3. Structured Instructions
Explicit formatting/tone guidance to reduce hallucinations.

---

# 📊 Evaluation Metrics Implemented

---

## 1. Fact Integration Score

Measures:
> Did the generated email include all required key facts?

Logic:
- Python keyword-based matching  
- No LLM involved  

---

## 2. Tone Consistency Score

Measures:
> Did the email maintain requested tone throughout?

Logic:
- Email split into:
    - Opening  
    - Body  
    - Closing  
- Each part scored independently.

---

## 3. Actionability Score

Measures:
> Does email clearly tell recipient what to do next?

Checks:
- Clear ask  
- Defined next step  
- Timeframe mentioned  

---

# 🤖 Models Compared

This project compares two LLMs:

### Gemma 3 27B (via Gemini)

### LLaMA 4 Scout (via Groq)

Both models are tested on the **same 10 scenarios**.

---

# 🚀 Setup Instructions (Run From Scratch)

Follow these steps carefully.

---

## Step 1 — Install Python

Make sure Python 3.10+ is installed.

Check version:

```bash
python --version
```

or

```bash
python3 --version
```

---

## Step 2 — Clone Repository

```bash
git clone https://github.com/Ravindra-Pagidala/email_assistant
cd email_assignment
```

---

## Step 3 — Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 4 — Activate Virtual Environment

### Mac/Linux:

```bash
source venv/bin/activate
```

### Windows:

```bash
venv\Scripts\activate
```

---

## Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install groq google-generativeai python-dotenv pandas requests httpx
```

---

# 🔐 Environment Variables Setup

Create a `.env` file in root directory.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## How to Get Keys

### Gemini Key:
Get from:
https://makersuite.google.com/app/apikey

---

### Groq Key:
Get from:
https://console.groq.com/keys

---

# ▶️ Running the Project

---

## Run Main App

```bash
python app.py
```

This will:

1. Load scenarios  
2. Generate emails  
3. Evaluate outputs  
4. Save CSV results  

---

# 📈 Generated Output

After successful execution:

You will see:

```bash
results/
├── gemini_results.csv
├── groq_results.csv
```

These CSVs contain:

- Raw generated emails  
- Metric scores  
- Average performance  

---

# 📌 Final Recommendation from Evaluation

Based on testing:

- **Gemma 3 27B** performed better overall in:
    - Tone consistency  
    - Actionability  
    - Reliability  

- **LLaMA 4 Scout** was significantly faster.

### Production Recommendation:
Use **Gemma 3 27B** when output quality matters more than speed.

---

# 👨‍💻 Author Notes

This project was designed to simulate how real-world LLM applications are built:

- Modular architecture  
- Prompt engineering separation  
- Evaluation pipeline  
- Multi-model benchmarking  
- Production decision analysis  

Rather than simply "calling an API", the goal was to demonstrate engineering thinking behind deploying LLM systems responsibly.

---
