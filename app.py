import streamlit as st
import json
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from src.rag_classifier import RAGTicketClassifier
from src.gemini_ai import (
    chat_with_ai,
    generate_ai_investigation
)
from src.database import (
    register_user,
    login_user,
    login_employee,
    save_ticket,
    load_history,
    get_all_users,
    get_all_tickets,
    delete_user,
    delete_ticket,
    update_ticket_status,
    change_password,
    get_all_employees,
    add_employee,
    delete_employee,
    update_employee_status,
    get_employee_name,
    get_employee_tickets,
    finish_ticket,
    add_notification,
    get_notifications,
    get_ticket_owner,
    assign_employee,
    save_feedback
)

st.set_page_config(
    page_title="AI Ticket Classification",
    page_icon="🎫",
    layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "employee_name" not in st.session_state:
    st.session_state.employee_name = ""

if "department" not in st.session_state:
    st.session_state.department = ""

if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_ticket" not in st.session_state:
    st.session_state.last_ticket = ""

if not st.session_state.logged_in:

    st.title("🔐 Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            user = login_user(username, password)

            if user:

                # Employee
                if user["role"] == "employee":

                    employee = login_employee(username, password)

                    if employee:

                        st.session_state.logged_in = True
                        st.session_state.username = employee["username"]
                        st.session_state.employee_name = employee["name"]
                        st.session_state.department = employee["department"]
                        st.session_state.role = "employee"

                        st.rerun()

                # Admin / Normal User
                else:

                    st.session_state.logged_in = True
                    st.session_state.username = user["username"]
                    st.session_state.role = user["role"]

                    st.rerun()

            st.error("Invalid username or password")

    with tab2:

        new_username = st.text_input(
            "New Username"
        )

        new_password = st.text_input(
            "New Password",
            type="password"
        )

        if st.button("Register"):

            if register_user(
                new_username,
                new_password
            ):

                st.success("Account created!")

            else:

                st.error("Username already exists.")

    st.stop()


# ==================================================================
# MAIN APP (only reachable after login)
# ==================================================================

@st.cache_resource
def load_classifier():
    classifier = RAGTicketClassifier(
        embedder_backend="sentence",
        top_k=5
    )

    classifier.load(
        "index_store",
        embedder_backend="sentence"
    )

    return classifier


with st.sidebar:

    st.title("🎫 AI Ticket")

    st.markdown("---")

    st.sidebar.title("Navigation")

    if st.session_state.get("role") == "employee":

        pages = [
            "👨‍💼 Employee Dashboard"
        ]

    else:

        pages = [
            "🏠 Home",
            "📊 Analytics",
            "📁 History",
            "👤 Profile",
            "🔔 Notifications",
            "⚙️ Settings",
            "🤖 AI Assistant"
        ]

        if st.session_state.get("role") == "admin":
            pages.append("🛠️ Admin")

    page = st.sidebar.radio(
        "",
        pages,
        key="nav_radio"
    )

    st.markdown("---")

    st.success("Backend Connected")

    st.info(f"👤 {st.session_state.username}")

    st.write("Embedding")

    st.code("Sentence Transformer")

    st.write("Vector DB")

    st.code("FAISS")

    st.write("LLM")

    st.code("GPT-4o-mini")

    st.markdown("---")

    if st.button("🚪 Logout", key="sidebar_logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.employee_name = ""
        st.session_state.department = ""

        st.rerun()

if page == "🏠 Home":

    classifier = load_classifier()

    st.title("🎫 AI Ticket Classification System")

    st.write("Smart RAG-based Support Ticket Analyzer")

    left, right = st.columns([1, 1])

    with left:

        st.subheader("📝 Enter Ticket")

        ticket = st.text_area(
            "",
            height=250,
            placeholder="Example:\nI was charged twice for my subscription..."
        )

        analyze = st.button(
            "🚀 Analyze Ticket",
            use_container_width=True
        )

    if analyze:
        if ticket.strip() == "":
            st.warning("Please enter a ticket.")
        else:
            result = classifier.classify(ticket)

            ai_report = generate_ai_investigation(
                ticket,
                result["category"],
                result["department"]
            )

            gemini_response = chat_with_ai(
                "Reply with exactly: Gemini Connected Successfully"
            )

            if gemini_response["success"]:
                st.success(gemini_response["text"])
            else:
                if "429" in gemini_response["text"]:
                    st.warning(
                        "⚠️ AI service is temporarily unavailable because the Gemini API quota has been reached. The system will continue using the local AI model."
                    )
                else:
                    st.error(gemini_response["text"])

            # Assign employee and get username
            assigned_username = assign_employee(result["department"])
            assigned_name = get_employee_name(assigned_username)

            result["assigned_to"] = assigned_username
            result["assigned_to_name"] = assigned_name

            st.session_state.last_result = result
            st.session_state.last_ticket = ticket

            with right:

                st.subheader("🤖 AI Analysis")

                # -----------------------------
                # Metrics
                # -----------------------------
                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Category", result["category"])
                    st.metric("Department", result["department"])
                    st.metric(
                        "Assigned Employee",
                        assigned_name
                    )

                with c2:
                    st.metric("Priority", result["priority"])
                    st.metric("ETA", result["resolution_time"])

                # -----------------------------
                # Confidence
                # -----------------------------
                confidence = result["confidence"]

                confidence = max(
                    0.0,
                    min(float(confidence), 1.0)
                )

                st.metric(
                    "Confidence",
                    f"{confidence*100:.1f}%"
                )

                st.progress(confidence)

                # -----------------------------
                # Similar Historical Cases (RAG)
                # -----------------------------
                st.subheader("📚 Similar Historical Cases")

                if result.get("historical_found", False):

                    st.info(
                        "The AI found a similar historical support ticket from the knowledge base."
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Similarity",
                            f"{result['similarity_score']:.1f}%"
                        )

                    with col2:
                        st.metric(
                            "Historical Category",
                            result["historical_category"]
                        )

                    st.write("### Previous Similar Ticket")

                    st.success(
                        result["historical_ticket"]
                    )

                    submit_anyway = st.checkbox(
                        "Submit this ticket anyway"
                    )

                    if not submit_anyway:
                        st.info("You can review the existing ticket instead of creating a duplicate.")
                        st.stop()

                else:

                    st.info(
                        "No similar historical cases were found."
                    )

                st.divider()

                # -----------------------------
                # Learned From Human Feedback
                # -----------------------------
                if result["used_feedback"]:

                    st.subheader("🧠 Learned From Human Feedback")

                    c1, c2 = st.columns(2)

                    with c1:

                        st.metric(
                            "Similarity",
                            f"{result['feedback_similarity']:.1f}%"
                        )

                    with c2:

                        st.metric(
                            "Corrected Category",
                            result["feedback_corrected"]
                        )

                    st.info(result["feedback_ticket"])

                    st.success(
                        "This prediction was improved using previous administrator feedback."
                    )

                # -----------------------------
                # AI Investigation
                # -----------------------------
                st.subheader("🧠 AI Investigation")

                if ai_report["success"]:
                    st.markdown(ai_report["text"])
                else:
                    if "429" in ai_report["text"]:
                        st.warning(
                            "⚠️ AI investigation is temporarily unavailable because the Gemini API quota has been reached. The system will continue using the local AI model."
                        )
                    else:
                        st.error(ai_report["text"])

                # Save ticket after duplicate check
                save_ticket(
                    st.session_state.username,
                    result,
                    ticket
                )

                # Send notifications
                add_notification(
                    st.session_state.username,
                    "Your ticket has been submitted successfully."
                )

                add_notification(
                    assigned_username,
                    "A new ticket has been assigned to you."
                )

                # -----------------------------
                # Sentiment
                # -----------------------------
                st.subheader("😊 AI Sentiment")

                sentiment = result["sentiment"]

                if sentiment == "Positive":
                    st.success(f"😊 {sentiment}")

                elif sentiment == "Negative":
                    st.error(f"😠 {sentiment}")

                else:
                    st.warning(f"😐 {sentiment}")

                st.caption(
                    f"AI Confidence: {result['sentiment_confidence']*100:.1f}%"
                )

                # -----------------------------
                # Urgency
                # -----------------------------
                st.subheader("🚨 Urgency")

                if result["urgency"] == "Critical":
                    st.error("🚨 Critical")

                elif result["urgency"] == "High":
                    st.warning("⚠️ High")

                else:
                    st.info("ℹ️ Normal")

                # -----------------------------
                # Summary
                # -----------------------------
                st.subheader("📝 Summary")

                st.info(result["summary"])

                # -----------------------------
                # Reason
                # -----------------------------
                st.subheader("📌 Reason")

                st.write(result["reason"])

                st.divider()

                # -----------------------------
                # Explainable AI
                # -----------------------------
                st.subheader("🧠 Explainable AI")

                exp = result["explanation"]

                st.success(
                    f"AI classified this ticket as **{result['category']}** "
                    f"because of the following evidence:"
                )

                st.write("### 🔑 Important Keywords")

                cols = st.columns(2)

                for i, item in enumerate(exp["keywords"]):

                    with cols[i % 2]:

                        st.metric(
                            item["keyword"],
                            f"{item['score']:.2f}"
                        )

                st.divider()

                st.write("### 📈 AI Decision Explanation")

                for reason in exp["reasons"]:

                    st.write(f"✅ {reason}")

                st.divider()

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Average Similarity",
                        f"{exp['average_similarity']:.1f}%"
                    )

                with c2:

                    st.metric(
                        "Nearest Neighbors",
                        f"{exp['agreement']}/{exp['total_neighbors']}"
                    )

                with c3:

                    st.metric(
                        "Dominant Category",
                        exp["dominant_category"]
                    )

                st.divider()

                # -----------------------------
                # Suggested Solution
                # -----------------------------
                st.subheader("✅ Suggested Solution")

                for step in result["solution"]:
                    st.success(step)

                st.divider()

                st.subheader("🤖 AI Suggested Reply")

                edited_reply = st.text_area(
                    "Edit Reply",
                    value=result["reply"],
                    height=180
                )

                st.code(
                    edited_reply,
                    language=None
                )

                # -----------------------------
                # Similar Tickets
                # -----------------------------
                st.subheader("📚 Similar Tickets")

                df = pd.DataFrame(result["evidence"])

                df = df.rename(
                    columns={
                        "category": "Category",
                        "similarity": "Similarity",
                        "text": "Ticket"
                    }
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    history = load_history(st.session_state.username)

    total = len(history)

    resolved = total
    open_tickets = 0

    if total > 0:
        avg_conf = sum(x["confidence"] for x in history) / total
    else:
        avg_conf = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tickets", total)

    with col2:
        st.metric("Resolved", resolved)

    with col3:
        st.metric("Open", open_tickets)

    with col4:
        st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Tickets by Category")

        category_counter = Counter()

        for item in history:
            category_counter[item["category"]] += 1

        if category_counter:
            st.bar_chart(category_counter)
        else:
            st.info("No data yet.")

    with col2:

        st.subheader("Priority Distribution")

        priority_counter = Counter()

        for item in history:
            priority_counter[item["priority"]] += 1

        if priority_counter:
            st.bar_chart(priority_counter)
        else:
            st.info("No data yet.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sentiment")

        sentiment_counter = Counter()

        for item in history:
            sentiment_counter[item["sentiment"]] += 1

        if sentiment_counter:
            st.bar_chart(sentiment_counter)
        else:
            st.info("No data yet.")

    with col2:

        st.subheader("Urgency")

        urgency_counter = Counter()

        for item in history:
            urgency_counter[item["urgency"]] += 1

        if urgency_counter:
            st.bar_chart(urgency_counter)
        else:
            st.info("No data yet.")

    st.divider()

    st.subheader("Recent Tickets")

    if history:

        df = pd.DataFrame(history)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No tickets analyzed yet.")

elif page == "👤 Profile":

    st.title("👤 My Profile")

    history = load_history(st.session_state.username)

    total = len(history)

    st.metric(
        "🎫 Total Tickets",
        total
    )

    st.write("### Username")

    st.success(st.session_state.username)

    if total > 0:

        avg = sum(
            x["confidence"] for x in history
        ) / total

        st.metric(
            "Average Confidence",
            f"{avg*100:.1f}%"
        )

    if total > 0:

        counter = Counter()

        for item in history:
            counter[item["category"]] += 1

        category = counter.most_common(1)[0]

        st.metric(
            "Most Used Category",
            category[0]
        )

    st.subheader("Recent Tickets")

    if total > 0:

        df = pd.DataFrame(history)

        st.dataframe(
            df.tail(5),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No tickets yet.")

elif page == "🔔 Notifications":

    st.title("🔔 Notifications")

    notifications = get_notifications(
        st.session_state.username
    )

    if not notifications:
        st.info("No notifications yet.")
    else:
        for n in notifications:
            st.info(f"{n[1]}  |  {n[0]}")

elif page == "🛠️ Admin":

    st.title("🛠️ Admin Dashboard")

    users = get_all_users()
    tickets = get_all_tickets()

    total_users = len(users)
    total_tickets = len(tickets)

    open_count = sum(
        1 for t in tickets
        if t[6] == "Open"
    )

    resolved_count = sum(
        1 for t in tickets
        if t[6] == "Resolved"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Users", total_users)
    col2.metric("🎫 Tickets", total_tickets)
    col3.metric("📂 Open", open_count)
    col4.metric("✅ Resolved", resolved_count)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👥 Users",
            "🎫 Tickets",
            "📊 Charts",
            "👥 Employees"
        ]
    )

    # ==========================
    # USERS
    # ==========================

    with tab1:

        st.subheader("Registered Users")

        df_users = pd.DataFrame(
            users,
            columns=["Username"]
        )

        st.dataframe(
            df_users,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        username = st.selectbox(
            "Delete User",
            df_users["Username"]
        )

        if st.button("🗑 Delete User", key="delete_user_btn"):

            delete_user(username)

            st.success("User deleted")

            st.rerun()

    # ==========================
    # TICKETS
    # ==========================

    with tab2:

        df = pd.DataFrame(
            tickets,
            columns=[
                "ID",
                "User",
                "Ticket",
                "Category",
                "Priority",
                "Department",
                "Assigned To",
                "Status",
                "Created"
            ]
        )

        # Convert assigned_to username to name for display
        df["Assigned To"] = df["Assigned To"].apply(
            lambda x: get_employee_name(x) if x else "Unassigned"
        )

        st.subheader("🔍 Search Tickets")

        keyword = st.text_input("Search by ticket text")

        filtered = df

        if keyword:
            filtered = filtered[
                filtered["Ticket"].str.contains(
                    keyword,
                    case=False,
                    na=False
                )
            ]

        category_filter = st.selectbox(
            "Category",
            ["All"] + sorted(df["Category"].unique().tolist())
        )

        if category_filter != "All":
            filtered = filtered[
                filtered["Category"] == category_filter
            ]

        priority_filter = st.selectbox(
            "Priority",
            ["All"] + sorted(df["Priority"].unique().tolist())
        )

        if priority_filter != "All":
            filtered = filtered[
                filtered["Priority"] == priority_filter
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("🧠 AI Feedback")

        if not filtered.empty:

            ticket_ids = filtered["ID"].tolist()

            selected_id = st.selectbox(
                "Select Ticket",
                ticket_ids
            )

            selected = filtered[
                filtered["ID"] == selected_id
            ].iloc[0]

            st.write("### Ticket")

            st.info(selected["Ticket"])

            st.write(
                "**AI Prediction:**",
                selected["Category"]
            )

            categories = [
                "Account Access",
                "Billing",
                "Technical Support",
                "General Inquiry"
            ]

            correct_category = st.selectbox(
                "Correct Category",
                categories,
                index=categories.index(selected["Category"])
                if selected["Category"] in categories else 0
            )

            if st.button("💾 Save Feedback"):

                save_feedback(
                    selected["Ticket"],
                    selected["Category"],
                    correct_category
                )

                st.success(
                    "Feedback saved. The AI will use this correction for similar tickets."
                )

        st.divider()

        st.subheader("Update Ticket Status")

        ticket_id = st.selectbox(
            "Ticket",
            df["ID"]
        )

        status = st.selectbox(
            "Status",
            [
                "Open",
                "In Progress",
                "Resolved"
            ]
        )

        if st.button(
            "Update",
            key="update_ticket_status"
        ):

            update_ticket_status(
                ticket_id,
                status
            )

            st.success("Updated")

            st.rerun()

        st.divider()

        delete_id = st.selectbox(
            "Delete Ticket",
            df["ID"],
            key="delete"
        )

        if st.button(
            "Delete Ticket",
            key="delete_ticket_btn"
        ):

            delete_ticket(delete_id)

            st.success("Deleted")

            st.rerun()

    # ==========================
    # CHARTS
    # ==========================

    with tab3:

        from collections import Counter

        # Most Active User
        user_counter = Counter()

        for t in tickets:
            user_counter[t[1]] += 1

        if user_counter:

            most_user = user_counter.most_common(1)[0]

            st.metric(
                "👑 Most Active User",
                most_user[0],
                f"{most_user[1]} Tickets"
            )

        # Most Common Category
        category_counter = Counter()

        for t in tickets:
            category_counter[t[3]] += 1

        if category_counter:

            most_category = category_counter.most_common(1)[0]

            st.metric(
                "🏆 Most Common Category",
                most_category[0],
                f"{most_category[1]} Tickets"
            )

        st.divider()

        # Bar Charts
        priority = Counter()
        status_counter = Counter()

        for t in tickets:
            priority[t[4]] += 1
            status_counter[t[6]] += 1

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("Tickets by Category")

            st.bar_chart(category_counter)

        with c2:

            st.subheader("Priority")

            st.bar_chart(priority)

        st.subheader("Ticket Status")

        st.bar_chart(status_counter)

        # Tickets Per Day
        st.divider()

        st.subheader("📅 Tickets Per Day")

        daily_counter = Counter()

        for t in tickets:

            day = str(t[7]).split(" ")[0]

            daily_counter[day] += 1

        st.line_chart(daily_counter)

        # Pie Charts
        st.divider()

        st.subheader("🥧 Category Distribution")

        fig, ax = plt.subplots()

        ax.pie(
            category_counter.values(),
            labels=category_counter.keys(),
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

        priority_counter = Counter()

        for t in tickets:
            priority_counter[t[4]] += 1

        st.subheader("🥧 Priority Distribution")

        fig2, ax2 = plt.subplots()

        ax2.pie(
            priority_counter.values(),
            labels=priority_counter.keys(),
            autopct="%1.1f%%"
        )

        st.pyplot(fig2)

    # ==========================
    # EMPLOYEES
    # ==========================

    with tab4:

        st.subheader("Employee Management")

        employees = get_all_employees()

        df = pd.DataFrame(
            employees,
            columns=[
                "ID",
                "Name",
                "Department",
                "Workload",
                "Status"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Add Employee")

        name = st.text_input("Employee Name")

        department = st.selectbox(
            "Department",
            [
                "Finance",
                "Technical Support",
                "Account",
                "Support"
            ]
        )

        if st.button(
            "Add Employee",
            key="add_employee_btn"
        ):

            add_employee(name, department)

            st.success("Employee Added")

            st.rerun()

        st.divider()

        st.subheader("Update Status")

        employee = st.selectbox(
            "Employee",
            df["ID"]
        )

        status = st.selectbox(
            "Status",
            [
                "Available",
                "Busy",
                "Offline"
            ]
        )

        if st.button(
            "Update",
            key="update_employee_status"
        ):

            update_employee_status(
                employee,
                status
            )

            st.success("Updated")

            st.rerun()

        st.divider()

        st.subheader("Delete Employee")

        emp = st.selectbox(
            "Delete",
            df["ID"],
            key="delete_employee"
        )

        if st.button(
            "Delete Employee",
            key="delete_employee_btn"
        ):

            delete_employee(emp)

            st.success("Deleted")

            st.rerun()

elif page == "👨‍💼 Employee Dashboard":

    st.title("👨‍💻 Employee Dashboard")

    st.success(
        f"Welcome {st.session_state.employee_name}"
    )

    st.write(f"Department: {st.session_state.department}")

    tickets = get_employee_tickets(st.session_state.username)

    if not tickets:
        st.info("No tickets assigned to you.")
    else:
        for ticket in tickets:
            with st.expander(
                f"Ticket #{ticket['id']} - {ticket['status']}"
            ):
                st.write(ticket["ticket"])
                st.write(f"Category: {ticket['category']}")
                st.write(f"Priority: {ticket['priority']}")

                status = st.selectbox(
                    "Status",
                    [
                        "Open",
                        "In Progress",
                        "Resolved"
                    ],
                    index=[
                        "Open",
                        "In Progress",
                        "Resolved"
                    ].index(ticket["status"]),
                    key=f"status_{ticket['id']}"
                )

                if st.button(
                    "Update Status",
                    key=f"btn_{ticket['id']}"
                ):
                    if status == "Resolved":
                        finish_ticket(ticket["id"])
                        customer = get_ticket_owner(ticket["id"])
                        add_notification(
                            customer,
                            "Your ticket has been resolved."
                        )
                    else:
                        update_ticket_status(ticket["id"], status)

                    st.success("Status Updated")
                    st.rerun()

elif page == "📁 History":

    st.title("📁 Ticket History")

    history = load_history(st.session_state.username)

    if len(history) == 0:
        st.info("No analyzed tickets yet.")
    else:

        st.subheader("Search")

        keyword = st.text_input(
            "Search Ticket"
        )

        filtered = []

        for item in history:

            if keyword.lower() in item["ticket"].lower():
                filtered.append(item)

        if keyword == "":
            filtered = history

        df = pd.DataFrame(filtered)

        df = df.rename(
            columns={
                "ticket": "Ticket",
                "category": "Category",
                "priority": "Priority",
                "department": "Department",
                "confidence": "Confidence",
                "sentiment": "Sentiment",
                "urgency": "Urgency"
            }
        )

        df["Confidence"] = (
            df["Confidence"] * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False)

        st.download_button(
            "📥 Download CSV",
            csv,
            "ticket_history.csv",
            "text/csv"
        )

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.subheader("🔒 Change Password")

    new_password = st.text_input(
        "New Password",
        type="password"
    )

    if st.button("Update Password"):

        if new_password.strip() == "":
            st.warning("Password cannot be empty.")

        else:

            change_password(
                st.session_state.username,
                new_password
            )

            st.success("Password updated successfully.")

    st.divider()

    if st.button("🚪 Logout", key="settings_logout"):

        st.session_state.clear()

        st.rerun()

elif page == "🤖 AI Assistant":

    st.title("🤖 AI Ticket Assistant")

    latest = st.session_state.last_result
    ticket = st.session_state.last_ticket

    if latest is None:
        st.info("Analyze a ticket first.")
        st.stop()

    st.divider()

    for msg in st.session_state.assistant_messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask me about your ticket..."
    )

    if question:

        st.session_state.assistant_messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        prompt = f"""
You are an AI customer support assistant.

Here is the analyzed ticket information:

Ticket:
{ticket}

Category:
{latest['category']}

Department:
{latest['department']}

Priority:
{latest['priority']}

Summary:
{latest['summary']}

Sentiment:
{latest['sentiment']}

The user asks:

{question}

Answer clearly and professionally.
"""

        answer = chat_with_ai(prompt)

        with st.chat_message("assistant"):
            if answer["success"]:
                st.markdown(answer["text"])
            else:
                if "429" in answer["text"]:
                    st.warning(
                        "⚠️ AI Assistant is temporarily unavailable because the Gemini API quota has been reached. Please try again later."
                    )
                else:
                    st.error(answer["text"])

        st.session_state.assistant_messages.append(
            {
                "role": "assistant",
                "content": answer["text"] if answer["success"] else "AI service unavailable."
            }
        )