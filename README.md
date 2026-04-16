# 🚀 AI Email Generation Assistant

A production-grade AI system that generates professional emails from structured inputs and evaluates model performance using custom metrics.

---

# 📌 What This Project Does

This project builds an **AI Email Generation Assistant** that:

* Takes structured input:

  * 🧠 Intent (why the email is being written)
  * 📌 Key Facts (must be included)
  * 🎯 Tone (formal / casual / urgent / empathetic)

* Generates a **high-quality professional email** using LLMs

* Evaluates output using **custom metrics**

* Compares multiple models (**Gemma vs LLaMA**) to determine:

  * Which produces better emails
  * Where models fail
  * Which is better for production

---

# 🧠 Key Features

### ✉️ Email Generation

* Advanced prompt engineering:

  * Role-based prompting
  * Few-shot examples
  * Structured reasoning (CoT)

---

### 📊 Evaluation System

Each generated email is scored on:

1. **Fact Integration**

   * Checks if all key facts are included

2. **Tone Consistency**

   * Ensures tone remains consistent (start → end)

3. **Actionability**

   * Checks if email clearly tells what to do next

---

### ⚙️ Production-Level Engineering

* 🔁 Retry with exponential backoff
* ⚡ Circuit Breaker (fail fast on API issues)
* 📜 Structured logging (JSON logs)
* 🔄 Fallback model support
* ⚡ Latency tracking
* 📁 CSV result storage

---

# 🏗️ Architecture Overview

```
User Input (Intent + Facts + Tone)
            ↓
   Prompt Builder (Advanced Prompting)
            ↓
   Email Generator Service
      ↓              ↓
  Gemini Client   Groq Client
      ↓              ↓
   Generated Emails (per model)
            ↓
     Evaluation Pipeline
            ↓
   Custom Metrics Scoring
            ↓
     CSV + Summary Output
```

---

# 📂 Project Structure

```
.
├── models/              # LLM clients (Gemini, Groq)
├── prompts/             # Advanced prompt engineering
├── evaluation/          # Metrics + evaluation pipeline
├── config/              # Logging, settings, circuit breaker
├── scenarios/           # Test scenarios (10 cases)
├── results/             # Output CSV files
├── run_evaluation.py    # Entry point
```

---

# ⚙️ Setup Instructions (Step-by-Step)

## 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd <repo-name>
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

### Windows:

```bash
venv\Scripts\activate
```

### Mac/Linux:

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is missing, install manually:

```bash
pip install google-generativeai
pip install groq
pip install python-dotenv
```

---

## 4️⃣ Setup Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Running the Project

### ✅ Run BOTH models (recommended)

```bash
python run_evaluation.py --model both
```

---

### ▶️ Run only Gemma (Gemini)

```bash
python run_evaluation.py --model gemma
```

---

### ▶️ Run only Groq (LLaMA)

```bash
python run_evaluation.py --model groq
```

---

# 📊 Output

After execution:

### 📁 CSV Files Generated

* `gemma_results.csv`
* `groq_results.csv`
* `comparison_summary.csv`
* partial result files

---

### 📈 Metrics Included

Each scenario contains:

* Fact Integration Score
* Tone Consistency Score
* Actionability Score
* Average Score
* Latency

---

# 🧪 Evaluation Design

* 10 real-world scenarios
* Each includes:

  * Intent
  * Key facts
  * Tone
  * Human reference email

---

# 📌 Model Comparison Summary

| Metric           | Gemma  | LLaMA          |
| ---------------- | ------ | -------------- |
| Fact Integration | 1.00   | 1.00           |
| Tone Consistency | Higher | Lower          |
| Actionability    | Higher | Slightly Lower |

### 🏆 Final Result:

👉 **Gemma performs better overall**

---

# ⚖️ Trade-offs

| Use Case                | Recommended Model |
| ----------------------- | ----------------- |
| Quality-focused systems | ✅ Gemma           |
| High-speed bulk systems | ⚡ LLaMA           |

---

# 🚀 Future Improvements

* Add semantic similarity vs reference emails
* Use independent judge model
* Improve urgent tone prompting
* Add UI (web interface)

---

# ⭐ Final Note

This project is not just a prototype —
it is designed with **production-grade engineering principles**.

---
# 👨‍💻 Author

Ravindra Pagidala

---
