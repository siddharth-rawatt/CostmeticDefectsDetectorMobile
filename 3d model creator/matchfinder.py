import sqlite3


def decode_pair_id(pair_id):
    id1 = pair_id >> 32
    id2 = pair_id & 0xFFFFFFFF
    return id1, id2


def get_image_pairs_with_matches(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT pair_id FROM two_view_geometries;")
    results = cursor.fetchall()
    conn.close()

    return [decode_pair_id(row[0]) for row in results]


def get_image_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, name FROM images;")
    id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return id_to_name


# Run it
db_path = "database.db"  # or full path if needed
pairs = get_image_pairs_with_matches(db_path)
id_to_name = get_image_names(db_path)

for id1, id2 in pairs:
    print(
        f"✅ Matched: {id_to_name[id1]} (ID {id1})  <-->  {id_to_name[id2]} (ID {id2})"
    )
