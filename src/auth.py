import sqlite3
import bcrypt

DB_NAME = "users.db"


def create_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT NOT NULL,

            ticket TEXT NOT NULL,

            category TEXT,

            priority TEXT,

            department TEXT,

            confidence REAL,

            sentiment TEXT,

            urgency TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password):

    return bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()


def verify_password(password, hashed):

    return bcrypt.checkpw(
        password.encode(),
        hashed.encode()
    )


def register_user(username, email, password):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users(username,email,password)
            VALUES(?,?,?)
            """,
            (
                username,
                email,
                hash_password(password)
            )
        )

        conn.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        conn.close()


def login_user(username, password):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT password
        FROM users
        WHERE username=?
        """,
        (username,)
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return False

    return verify_password(
        password,
        row[0]
    )


def save_ticket(
    username,
    ticket,
    category,
    priority,
    department,
    confidence,
    sentiment,
    urgency
):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO ticket_history(
            username,
            ticket,
            category,
            priority,
            department,
            confidence,
            sentiment,
            urgency
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        ticket,
        category,
        priority,
        department,
        confidence,
        sentiment,
        urgency
    ))

    conn.commit()
    conn.close()


def load_history(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            ticket,
            category,
            priority,
            department,
            confidence,
            sentiment,
            urgency,
            created_at
        FROM ticket_history
        WHERE username=?
        ORDER BY created_at DESC
    """, (username,))

    rows = cursor.fetchall()

    conn.close()

    return rows