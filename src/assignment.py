import random

EMPLOYEES = {

    "Billing": [
        "Ahmed Hassan",
        "Mohamed Ali",
        "Sara Adel"
    ],

    "Technical Support": [
        "Omar Khaled",
        "Youssef Tarek",
        "Mariam Ahmed"
    ],

    "Account": [
        "Nour Mohamed",
        "Fatma Ali"
    ],

    "Sales": [
        "Mahmoud Samir",
        "Salma Hassan"
    ],

    "Support": [
        "General Support Team"
    ]

}


def assign_employee(department):

    employees = EMPLOYEES.get(
        department,
        ["General Support Team"]
    )

    return random.choice(employees)