import sqlite3
from datetime import datetime

DB_NAME = "users.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def make_admin(username):
    conn= get_connection()
    cursor= conn.cursor()
    cursor.execute("UPDATE users SET role='admin' WHERE username=?",(username,))  

    conn.commit()
    conn.close()  


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        name TEXT,
        department TEXT,
        workload INTEGER DEFAULT 0,
        status TEXT DEFAULT 'Available'
    )
    """)

    employees = [
        ("ahmed", "1234", "Ahmed Hassan", "Finance"),
        ("mohamed", "1234", "Mohamed Ali", "Finance"),
        ("omar", "1234", "Omar Khaled", "Technical Support"),
        ("mariam", "1234", "Mariam Ahmed", "Technical Support"),
        ("nour", "1234", "Nour Mohamed", "Account"),
        ("sara", "1234", "Sara Adel", "Account"),
        ("support", "1234", "General Support", "Support")
    ]

    for emp in employees:
        cursor.execute("""
        INSERT OR IGNORE INTO employees(
        username,
        password,
        name,
        department
        )
        VALUES(?,?,?,?)
        """, emp)

    # Create employee login accounts
    employee_accounts = [
        ("ahmed", "1234", "employee"),
        ("mohamed", "1234", "employee"),
        ("omar", "1234", "employee"),
        ("mariam", "1234", "employee"),
        ("nour", "1234", "employee"),
        ("sara", "1234", "employee"),
        ("support", "1234", "employee")
    ]

    for account in employee_accounts:
        cursor.execute("""
        INSERT OR IGNORE INTO users(username,password,role)
        VALUES(?,?,?)
        """, account)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        ticket TEXT,
        category TEXT,
        priority TEXT,
        department TEXT,
        assigned_to TEXT,
        confidence REAL,
        sentiment TEXT,
        urgency TEXT,
        status TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket TEXT,
        predicted_category TEXT,
        corrected_category TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notifications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        message TEXT,
        created_at TEXT,
        is_read INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()

    # Create default admin user
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR IGNORE INTO users(username,password,role)
    VALUES(?,?,?)
    """, ("admin", "admin123", "admin"))
    conn.commit()
    conn.close()


def register_user(username, password):
    if username == "admin":
        return False

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users(username,password,role) VALUES(?,?,?)",
            (username, password, "user")
        )
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False


def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT username, password, role
        FROM users
        WHERE username=?
        """,
        (username,)
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    if row["password"] != password:
        return None

    return {
        "username": row["username"],
        "role": row["role"]
    }


def login_employee(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT username, name, department
    FROM employees
    WHERE username=? AND password=?
    """, (username, password))

    row = cursor.fetchone()

    conn.close()

    if row:
        return {
            "username": row["username"],
            "name": row["name"],
            "department": row["department"]
        }

    return None


def save_ticket(username, result, ticket):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO tickets
    (
    username,
    ticket,
    category,
    priority,
    department,
    assigned_to,
    confidence,
    sentiment,
    urgency,
    status,
    created_at
    )
    VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        username,
        ticket,
        result["category"],
        result["priority"],
        result["department"],
        result["assigned_to"],
        result["confidence"],
        result["sentiment"],
        result["urgency"],
        "Open",
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))

    conn.commit()
    conn.close()


def load_history(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT ticket,category,priority,department,confidence,sentiment,urgency,status,created_at FROM tickets WHERE username=?",
        (username,)
    )

    rows = cursor.fetchall()
    conn.close()

    history = []

    for row in rows:
        history.append({
            "ticket": row[0],
            "category": row[1],
            "priority": row[2],
            "department": row[3],
            "confidence": row[4],
            "sentiment": row[5],
            "urgency": row[6],
            "status": row[7],
            "created_at": row[8]
        })

    return history


def get_all_users():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT username
    FROM users
    ORDER BY username
    """)

    users = cursor.fetchall()

    conn.close()

    return users


def get_all_tickets():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        username,
        ticket,
        category,
        priority,
        department,
        assigned_to,
        status,
        created_at
    FROM tickets
    ORDER BY id DESC
    """)

    tickets = cursor.fetchall()

    conn.close()

    return tickets


def delete_ticket(ticket_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM tickets WHERE id=?",
        (ticket_id,)
    )

    conn.commit()
    conn.close()


def update_ticket_status(ticket_id, status):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE tickets
        SET status=?
        WHERE id=?
        """,
        (status, ticket_id)
    )

    conn.commit()
    conn.close()


def finish_ticket(ticket_id):
    conn = get_connection()
    cursor = conn.cursor()

    # معرفة الموظف المسؤول
    cursor.execute("""
    SELECT assigned_to
    FROM tickets
    WHERE id=?
    """, (ticket_id,))

    row = cursor.fetchone()

    if row:

        employee_username = row["assigned_to"]

        # تحديث حالة التذكرة
        cursor.execute("""
        UPDATE tickets
        SET status='Resolved'
        WHERE id=?
        """, (ticket_id,))

        # تقليل الـ workload
        cursor.execute("""
        UPDATE employees
        SET workload = workload - 1
        WHERE username=?
        AND workload > 0
        """, (employee_username,))

    conn.commit()
    conn.close()


def get_ticket_owner(ticket_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT username
    FROM tickets
    WHERE id=?
    """, (ticket_id,))

    row = cursor.fetchone()

    conn.close()

    if row:
        return row["username"]

    return None


def delete_user(username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM users WHERE username=?",
        (username,)
    )

    cursor.execute(
        "DELETE FROM tickets WHERE username=?",
        (username,)
    )

    conn.commit()
    conn.close()


def change_password(username, new_password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET password=?
        WHERE username=?
        """,
        (new_password, username)
    )

    conn.commit()
    conn.close()


def assign_employee(department):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, username, name
        FROM employees
        WHERE department=?
        AND status='Available'
        ORDER BY workload ASC
        LIMIT 1
        """,
        (department,)
    )

    employee = cursor.fetchone()

    if employee is None:

        cursor.execute("""
        SELECT id, username, name
        FROM employees
        ORDER BY workload ASC
        LIMIT 1
        """)

        employee = cursor.fetchone()

    cursor.execute(
        """
        UPDATE employees
        SET workload = workload + 1
        WHERE id=?
        """,
        (employee["id"],)
    )

    conn.commit()
    conn.close()

    return employee["username"]


def get_all_employees():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id,name,department,workload,status
    FROM employees
    ORDER BY department,name
    """)

    employees = cursor.fetchall()

    conn.close()

    return employees


def add_employee(name, department):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO employees(
    username,
    password,
    name,
    department
    )
    VALUES(?,?,?,?)
    """, (name, name, name, department))

    conn.commit()
    conn.close()


def delete_employee(employee_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM employees WHERE id=?",
        (employee_id,)
    )

    conn.commit()
    conn.close()


def update_employee_status(employee_id, status):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE employees
    SET status=?
    WHERE id=?
    """, (status, employee_id))

    conn.commit()
    conn.close()


def get_employee_tickets(employee_username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        ticket,
        category,
        priority,
        status,
        created_at
    FROM tickets
    WHERE assigned_to=?
    ORDER BY id DESC
    """, (employee_username,))

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "id": row["id"],
            "ticket": row["ticket"],
            "category": row["category"],
            "priority": row["priority"],
            "status": row["status"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]


def get_employee_name(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT name
    FROM employees
    WHERE username=?
    """, (username,))

    row = cursor.fetchone()

    conn.close()

    if row:
        return row["name"]

    return None


def add_notification(username, message):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO notifications(
        username,
        message,
        created_at
    )
    VALUES(?,?,?)
    """, (
        username,
        message,
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))

    conn.commit()
    conn.close()


def get_notifications(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT message,created_at
    FROM notifications
    WHERE username=?
    ORDER BY id DESC
    """,(username,))

    rows=cursor.fetchall()

    conn.close()

    return rows


def save_feedback(ticket, predicted, corrected):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO feedback(
        ticket,
        predicted_category,
        corrected_category,
        created_at
    )
    VALUES(?,?,?,?)
    """, (
        ticket,
        predicted,
        corrected,
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))

    conn.commit()
    conn.close()


def get_feedback():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM feedback
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_feedback_tickets():
    rows = get_feedback()

    return [
        {
            "ticket": row[1],
            "predicted": row[2],
            "corrected": row[3]
        }
        for row in rows
    ]


def search_feedback(ticket):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM feedback
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


create_tables()