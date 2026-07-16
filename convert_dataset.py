import pandas as pd

# اقرأ ملف Kaggle
df = pd.read_csv("data/customer_support_tickets.csv")

# اختار الأعمدة المهمة
new_df = pd.DataFrame()

new_df["text"] = df["Ticket Description"]
new_df["category"] = df["Ticket Type"]
new_df["priority"] = df["Ticket Priority"]
new_df["resolution"] = df["Resolution"]
new_df["resolution_time"] = df["Time to Resolution"]

# حذف الصفوف الفارغة
new_df = new_df.dropna()

# حذف التكرار
new_df = new_df.drop_duplicates(subset=["text"])

# حفظ الملف الجديد
new_df.to_csv(
    "data/historical_tickets_new.csv",
    index=False
)

print("=" * 50)
print("Dataset Converted Successfully!")
print(f"Total Tickets: {len(new_df)}")
print("=" * 50)