#!/usr/bin/env python3
"""
match_resumes_jds.py

Reads:
  - resumes_samples.csv (columns: resume_id, Category, Text)
  - job_desc_sampled.csv (columns: jd_id, job_title, job_description)

Writes:
  - resume_jd_scores.csv (columns: resume_id, jd_id, score_json)

Usage:
  pip install openai pandas
  export OPENAI_API_KEY="your_api_key_here"
  python match_resumes_jds.py
"""

import os
import time
import json
import pandas as pd
import openai

# --- Configuration ---
RESUMES_CSV = "resumes_samples.csv"
JDS_CSV     = "job_desc_sampled.csv"
OUTPUT_CSV  = "resume_jd_scores.csv"
MODEL       = "gpt-3.5-turbo"   # cheapest ChatCompletion model
DELAY_SEC   = 0.5               # pause between requests to avoid rate limits

# --- Rubric-based scoring prompt ---
PROMPT_TEMPLATE = 
"""You are given a candidate resume and a job description.
Use the following weighted rubric to score their match:

- Role Alignment (30%)
- Skills Match (35%)
- Experience Fit (20%)
- Project Relevance (10%)
- Education/Certifications (5%)

For each category, provide a score between 0-100%. Then compute the weighted overall_match_percentage.
Return ONLY a JSON object in this exact format:

{{
  "overall_match_percentage": "X%",
  "score_breakdown": {{
    "role_alignment": {{ "score": "X%" }},
    "skills_match": {{ "score": "X%" }},
    "experience_match": {{ "score": "X%" }},
    "project_relevance": {{ "score": "X%" }},
    "education_match": {{ "score": "X%" }}
  }}
}}

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{jd_text}\"\"\"
"""

def compute_match(resume_text: str, jd_text: str) -> dict:
    """Call OpenAI ChatCompletion to compute the rubric scores."""
    prompt = PROMPT_TEMPLATE.format(resume_text=resume_text, jd_text=jd_text)
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert recruiter and resume matcher."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    return json.loads(content)

def main():
    # Load API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    # Read inputs
    resumes = pd.read_csv(RESUMES_CSV)
    jds     = pd.read_csv(JDS_CSV)

    results = []
    total = len(resumes) * len(jds)
    counter = 0

    # Compute for each resume–JD pair
    for _, r in resumes.iterrows():
        for _, jd in jds.iterrows():
            counter += 1
            print(f"[{counter}/{total}] Matching {r['resume_id']} → {jd['jd_id']}...")
            try:
                score = compute_match(r['Text'], jd['job_description'])
            except Exception as e:
                print(f"  ⚠️ Error on {r['resume_id']},{jd['jd_id']}: {e}")
                score = {
                    "overall_match_percentage": None,
                    "score_breakdown": {}
                }
            results.append({
                "resume_id": r['resume_id'],
                "jd_id":     jd['jd_id'],
                "score_json": json.dumps(score)
            })
            time.sleep(DELAY_SEC)

    # Save output
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! Results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
