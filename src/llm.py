from google import genai
import os
import time
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_segment_narrative(segment_row, total_customers=3920):
    prompt = f"""
You are a senior customer analytics consultant presenting to a C-suite audience.
You have deep expertise in RFM analysis, customer lifetime value, and retention strategy.

You are analyzing a segment from a UK-based e-commerce retailer (gifting/homewares).
The segmentation was built using K-Means clustering on log-transformed, standardized
RFM features derived from 349,203 transactions across 3,920 customers over 12 months.

SEGMENT DATA:
- Segment Name: {segment_row["cluster_label"]}
- Segment Size: {segment_row["num_customers"]} customers ({round(segment_row["num_customers"]/total_customers*100, 1)}% of base)
- Avg Recency: {segment_row["avg_recency"]} days since last purchase
- Avg Frequency: {segment_row["avg_frequency"]} orders per customer
- Avg Monetary Value: £{segment_row["avg_monetary"]} lifetime spend

Provide a sharp, data-driven report in exactly 4 sections:

1. SEGMENT PROFILE
   Characterize this segment using the RFM metrics above. Reference the specific
   numbers. Explain what the combination of recency, frequency, and monetary value
   tells us about their purchasing behavior and relationship with the brand.

2. REVENUE IMPACT
   Quantify the revenue opportunity or risk. Calculate the estimated total revenue
   this segment represents (size x monetary). Compare it relative to other segments
   if relevant. Be precise.

3. CHURN / RETENTION RISK
   Assess the retention risk using recency as the primary signal. Reference
   industry benchmarks where relevant (e.g. e-commerce average purchase cycle).
   Be direct about whether this segment is deteriorating, stable, or growing.

4. RECOMMENDED ACTIONS
   Provide exactly 3 highly specific, actionable recommendations tailored to this
   segment's RFM profile. Each action must include:
   - The specific tactic
   - The channel (email, SMS, paid retargeting, etc.)
   - The expected outcome tied to an RFM metric

Tone: executive, data-driven, no filler phrases. Every sentence must add value.
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text


def generate_all_narratives(segment_summary_df, total_customers=3920, sleep=1):
    narratives = {}
    for _, row in segment_summary_df.iterrows():
        narratives[row["cluster_label"]] = generate_segment_narrative(
            row, total_customers
        )
        time.sleep(sleep)
    return narratives
