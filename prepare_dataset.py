import random
import re
import pandas as pd

# ============================
# Load Dataset
# ============================

df = pd.read_csv("data/historical_tickets.csv")

# ============================
# Product Names
# ============================

products = [
    "Dell Laptop",
    "HP Printer",
    "Canon Camera",
    "Samsung Galaxy Phone",
    "iPhone",
    "MacBook Pro",
    "Lenovo ThinkPad",
    "LG Smart TV",
    "Sony Headphones",
    "Microsoft Surface",
    "Asus Router",
    "Cisco Switch",
    "Windows PC",
    "Office 365",
    "Outlook"
]

# ============================
# Category Mapping
# ============================

CATEGORY_MAP = {
    "Technical issue": "Technical Support",
    "Technical Issue": "Technical Support",
    "Product inquiry": "General Inquiry",
    "Product Inquiry": "General Inquiry",
    "Refund request": "Billing",
    "Cancellation request": "Account Management",
    "Billing inquiry": "Billing",
    "Payment problem": "Billing",
    "Shipping issue": "Customer Service"
}

# ============================
# Department Mapping
# ============================

DEPARTMENT_MAP = {
    "Technical Support": "IT Support",
    "Billing": "Finance",
    "Account Management": "Accounts",
    "Customer Service": "Support",
    "General Inquiry": "Support"
}

# ============================
# Default Resolutions
# ============================

DEFAULT_RESOLUTION = {
    "Technical Support":
        "Restart the device. Update the software. Contact IT if the issue persists.",

    "Billing":
        "Verify payment information. Review transaction history. Contact Finance.",

    "Account Management":
        "Verify account ownership. Process account request. Notify customer.",

    "Customer Service":
        "Review customer request. Contact customer if more information is required.",

    "General Inquiry":
        "Provide the requested information to the customer."
}

# ============================
# Resolution Time
# ============================

ETA = {
    "Technical Support": "2 Hours",
    "Billing": "24 Hours",
    "Account Management": "4 Hours",
    "Customer Service": "8 Hours",
    "General Inquiry": "1 Hour"
}

# ============================
# Clean Ticket Text
# ============================

def clean_text(text):

    text = str(text)

    text = text.replace(
        "{product_purchased}",
        random.choice(products)
    )

    text = text.replace(
        "{customer_name}",
        "Customer"
    )

    text = text.replace(
        "{ticket_id}",
        ""
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ============================
# Apply Cleaning
# ============================

df["text"] = df["text"].apply(clean_text)

# Remove duplicates
df = df.drop_duplicates(subset=["text"])

df = df.dropna(subset=["text"])

# ============================
# Standardize Categories
# ============================

df["category"] = (
    df["category"]
    .replace(CATEGORY_MAP)
)

# ============================
# Department
# ============================

df["department"] = (
    df["category"]
    .map(DEPARTMENT_MAP)
)

# ============================
# Resolution
# ============================

df["resolution"] = df.apply(
    lambda row:
    DEFAULT_RESOLUTION.get(
        row["category"],
        "Investigate the issue."
    ),
    axis=1
)

# ============================
# ETA
# ============================

df["resolution_time"] = (
    df["category"]
    .map(ETA)
)

# ============================
# Ticket Type
# ============================

df["ticket_type"] = "Incident"

# ============================
# Status
# ============================

df["status"] = "Closed"

# ============================
# Keywords
# ============================

def extract_keywords(text):

    words = text.lower().split()

    words = [
        w for w in words
        if len(w) > 4
    ]

    return ", ".join(words[:5])

df["keywords"] = df["text"].apply(
    extract_keywords
)

# ============================
# Save
# ============================

df.to_csv(
    "data/historical_tickets_clean.csv",
    index=False
)

print("=" * 60)
print("Dataset Prepared Successfully!")
print("Tickets :", len(df))
print("Columns :", list(df.columns))
print("=" * 60)