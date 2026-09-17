import sqlite3

conn = sqlite3.connect("data/observatory.db")
cursor = conn.cursor()

# Check inserted post
cursor.execute("SELECT post_id, channel_id, text FROM telegram_posts;")
print("Saved Posts:", cursor.fetchall())

# Check inserted prediction & abstention record
cursor.execute("SELECT post_id, status, max_confidence, requires_human_adjudication FROM model_predictions;")
print("Saved Predictions:", cursor.fetchall())

conn.close()