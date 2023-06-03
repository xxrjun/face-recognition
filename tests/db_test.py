import sqlite3
import numpy as np
import io


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
conn_db = sqlite3.connect('../database/database.db')

cursor = conn_db.execute("SELECT * FROM face_info")
db_data = cursor.fetchall()
for data in db_data:
    print(data)

conn_db.close()