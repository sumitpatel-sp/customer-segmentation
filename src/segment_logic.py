def get_segment_info(income, spending):

    if income >= 50 and spending >= 50:
        return {
            "name": "High Income - High Spending",
            "description": "Valuable customers already spending a lot",
            "strategy": "Loyalty rewards, premium membership, early product access"
        }

    elif income >= 50 and spending < 50:
        return {
            "name": "High Income - Low Spending",
            "description": "Have money but not spending much",
            "strategy": "Upsell, targeted marketing, premium recommendations"
        }

    elif income < 50 and spending >= 50:
        return {
            "name": "Low Income - High Spending",
            "description": "Spending a lot but budget sensitive",
            "strategy": "Discounts, bundles, retarget offers"
        }

    else:
        return {
            "name": "Low Income - Low Spending",
            "description": "Low value customers",
            "strategy": "Low cost campaigns, awareness campaigns"
        }