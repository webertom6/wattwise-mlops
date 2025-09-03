import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Connect to DB
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)

cursor = conn.cursor()

# Drop the predictions table
cursor.execute("DROP TABLE IF EXISTS predictions;")

conn.commit()
cursor.close()
conn.close()
print("🗑️ Table 'predictions' dropped successfully.")
