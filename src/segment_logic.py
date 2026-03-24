"""
RFM-based segment logic.

Segments
--------
High Value  — High spenders who buy frequently
Loyal       — Frequent buyers but moderate spend
At Risk     — Haven't purchased recently
Potential Loyalists — Frequent buyers with moderate spending
"""


def label_segment(recency: float, frequency: float, monetary: float) -> str:
    """Return a segment label from raw (unscaled) RFM values."""
    if monetary > 1000 and frequency > 10:
        return "High Value"
    elif frequency > 10:
        return "Loyal"
    elif recency > 100:
        return "At Risk"
    else:
        return "Potential Loyalists"


SEGMENT_INFO = {
    "High Value": {
        "name":        "High Value",
        "description": "High-spend, high-frequency, recent buyers — your most profitable customers",
        "strategy":    "VIP programmes, exclusive early access, personal account managers",
        "goal":        "Retain & delight — maximise lifetime value",
    },
    "Loyal": {
        "name":        "Loyal",
        "description": "Frequent buyers with moderate spend — consistent and reliable",
        "strategy":    "Cross-sell, volume discounts, subscription incentives",
        "goal":        "Grow basket size and increase spend per order",
    },
    "At Risk": {
        "name":        "At Risk",
        "description": "Customers who haven't purchased recently — potential churners",
        "strategy":    "Win-back campaigns, personalised outreach, special discounts",
        "goal":        "Re-engage before permanent churn",
    },
    "Potential Loyalists": {
        "name":        "Potential Loyalists",
        "description": "Frequent buyers with moderate spending — strong potential to become high-value customers",
        "strategy":    "Personalized offers, loyalty programs, upselling & cross-selling strategies",
        "goal":        "Convert into high-value customers through targeted engagement",
    },
}


def get_segment_info(recency: float, frequency: float, monetary: float) -> dict:
    """Return the full segment info dict for the given raw RFM values."""
    seg = label_segment(recency, frequency, monetary)
    return SEGMENT_INFO[seg]


def get_segment_info_by_name(name: str) -> dict:
    """Return description & strategy for a segment name."""
    return SEGMENT_INFO.get(name, {"description": "", "strategy": "", "goal": ""})