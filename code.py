import streamlit as st
import google.generativeai as genai
import uuid
from datetime import datetime
import time
import random

# --- Configuration & Initialization ---
st.set_page_config(layout="wide", page_title="Intelligent Queue System (Gemini)")

# Initialize session state variables if they don't exist
if 'initialized' not in st.session_state:
    st.session_state.gemini_api_key = "" # Changed from openai_api_key
    st.session_state.queue = []  # List of dictionaries for queue items
    st.session_state.agents = [] # List of dictionaries for agents
    st.session_state.next_agent_id_counter = 1
    st.session_state.logs = [] # For system logs/events
    st.session_state.initialized = True

# --- Helper Functions ---

def log_event(event_message):
    """Adds an event to the system log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {event_message}")
    if len(st.session_state.logs) > 20: # Keep logs to a reasonable size
        st.session_state.logs.pop(0)

def get_gemini_response(prompt_text, temperature=0.7, max_tokens=250): # Renamed and updated
    """Generates a response from Google's Gemini model."""
    if not st.session_state.gemini_api_key:
        st.error("Google AI API Key not set. Please set it in the sidebar.")
        return None
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # For Gemini, we can define a system-like instruction within the prompt itself
        # or use multi-turn chat if needed for more complex context.
        # For simple requests, a direct prompt is often sufficient.
        # You might prepend a general instruction like:
        # "You are a helpful assistant for a customer support queue. "
        # However, for specific tasks like summarization or response suggestion,
        # the prompt_text itself will carry the main instruction.

        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using Gemini 1.5 Flash
        
        full_prompt = f"As a helpful assistant for a customer support queue: {prompt_text}"

        response = model.generate_content(full_prompt, generation_config=generation_config)
        
        # Handle potential blocks or empty responses
        if response.parts:
            return response.text
        else:
            # This can happen if the response was blocked due to safety settings
            # or if no content was generated.
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            st.warning(f"Gemini AI: No content generated or response blocked. Reason: {block_reason}")
            log_event(f"Gemini AI: No content or blocked. Reason: {block_reason} for prompt: {prompt_text[:50]}...")
            return None

    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        log_event(f"Gemini API Error: {e}")
        return None

def add_to_queue(user_name, issue_description, priority="Medium"):
    """Adds a new item to the support queue."""
    item_id = str(uuid.uuid4())
    new_item = {
        "id": item_id,
        "user_name": user_name,
        "issue_description": issue_description,
        "priority": priority, # "Low", "Medium", "High"
        "timestamp": datetime.now(),
        "status": "Pending", # Pending, Assigned, In Progress, Resolved
        "assigned_agent_id": None,
        "resolution_notes": "",
        "suggested_priority_reason": "" # For agentic AI
    }
    # Agentic AI: Basic Priority Suggestion based on keywords
    if "urgent" in issue_description.lower() or "critical" in issue_description.lower():
        new_item["priority"] = "High"
        new_item["suggested_priority_reason"] = "Keyword 'urgent' or 'critical' detected."
    elif "asap" in issue_description.lower():
        new_item["priority"] = "High"
        new_item["suggested_priority_reason"] = "Keyword 'ASAP' detected."

    st.session_state.queue.append(new_item)
    log_event(f"New item added to queue: ID {item_id[:8]} by {user_name}, Priority: {new_item['priority']}")
    return item_id

def add_agent(agent_name):
    """Adds a new agent to the system."""
    agent_id = f"agent_{st.session_state.next_agent_id_counter}"
    st.session_state.next_agent_id_counter += 1
    new_agent = {
        "id": agent_id,
        "name": agent_name,
        "available": True,
        "current_task_id": None,
        "tasks_resolved": 0,
        "specialization": random.choice(["General", "Technical", "Billing", "Product"])
    }
    st.session_state.agents.append(new_agent)
    log_event(f"New agent added: {agent_name} (ID: {agent_id})")
    return agent_id

def assign_task_to_agent(task_id, agent_id):
    """Assigns a task to an agent."""
    task = next((t for t in st.session_state.queue if t["id"] == task_id), None)
    agent = next((a for a in st.session_state.agents if a["id"] == agent_id), None)

    if task and agent and agent["available"] and task["status"] == "Pending":
        task["status"] = "Assigned"
        task["assigned_agent_id"] = agent_id
        agent["available"] = False
        agent["current_task_id"] = task_id
        log_event(f"Task {task_id[:8]} assigned to agent {agent['name']}.")
        return True
    log_event(f"Failed to assign task {task_id[:8]} to agent {agent_id if agent else 'N/A'}.")
    return False

def resolve_task(task_id, resolution_notes):
    """Marks a task as resolved."""
    task = next((t for t in st.session_state.queue if t["id"] == task_id), None)
    if task and task["assigned_agent_id"]:
        agent = next((a for a in st.session_state.agents if a["id"] == task["assigned_agent_id"]), None)
        task["status"] = "Resolved"
        task["resolution_notes"] = resolution_notes
        if agent:
            agent["available"] = True
            agent["current_task_id"] = None
            agent["tasks_resolved"] += 1
        log_event(f"Task {task_id[:8]} resolved by agent {agent['name'] if agent else 'Unknown'}.")
        return True
    log_event(f"Failed to resolve task {task_id[:8]}. Task not found or not assigned.")
    return False

def get_suggested_next_task(agent_id=None):
    """Agentic AI: Suggests the next task to pick based on priority and age."""
    pending_tasks = [t for t in st.session_state.queue if t["status"] == "Pending"]
    if not pending_tasks:
        return None

    priority_map = {"High": 0, "Medium": 1, "Low": 2}
    pending_tasks.sort(key=lambda t: (priority_map.get(t["priority"], 3), t["timestamp"]))
    
    return pending_tasks[0]


# --- Streamlit UI ---

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    # Changed to Gemini API Key
    st.session_state.gemini_api_key = st.text_input(
        "Google AI Studio API Key", 
        type="password", 
        value=st.session_state.gemini_api_key,
        help="Get your key from Google AI Studio (makersuite.google.com)"
    )
    if not st.session_state.gemini_api_key:
        st.warning("Please enter your Google AI API Key to enable Generative AI features.")

    st.subheader("👤 Add Agent")
    with st.form("add_agent_form"):
        new_agent_name = st.text_input("Agent Name")
        submitted_add_agent = st.form_submit_button("Add Agent")
        if submitted_add_agent and new_agent_name:
            add_agent(new_agent_name)
            st.success(f"Agent {new_agent_name} added.")

    if not st.session_state.agents:
        if st.button("Add Default Agents (Alice, Bob)"):
            add_agent("Alice")
            add_agent("Bob")
            st.rerun()

    st.subheader("📊 System Stats")
    st.metric("Total Tasks in Queue", len(st.session_state.queue))
    st.metric("Pending Tasks", len([t for t in st.session_state.queue if t["status"] == "Pending"]))
    st.metric("Total Agents", len(st.session_state.agents))
    st.metric("Available Agents", len([a for a in st.session_state.agents if a["available"]]))

    st.subheader("📜 System Logs")
    log_container = st.container(height=200)
    for log in reversed(st.session_state.logs):
        log_container.caption(log)

# --- Main Application Area ---
st.title("🚀 Intelligent Queue Management System (Powered by Gemini)")
st.markdown("Manage customer requests efficiently with AI-powered assistance.")

tab1, tab2, tab3 = st.tabs(["📥 Submit Request", "📊 Queue Dashboard", "🧑‍💻 Agent View"])

# --- Tab 1: Submit Request ---
with tab1:
    st.header("➕ Submit a New Request")
    with st.form("new_request_form", clear_on_submit=True):
        user_name = st.text_input("Your Name", placeholder="e.g., John Doe")
        issue_description = st.text_area("Describe your issue", placeholder="e.g., My internet is not working.")
        priority_options = ["Low", "Medium", "High"]
        priority = st.selectbox("Priority (can be auto-adjusted by AI)", priority_options, index=1)
        submitted_request = st.form_submit_button("Submit Request")

        if submitted_request:
            if user_name and issue_description:
                item_id = add_to_queue(user_name, issue_description, priority)
                st.success(f"Request submitted successfully! Your Ticket ID: {item_id[:8]}")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Please fill in all fields.")

# --- Tab 2: Queue Dashboard ---
with tab2:
    st.header("📋 Current Queue")
    if not st.session_state.queue:
        st.info("The queue is currently empty.")
    else:
        queue_data = []
        for item in sorted(st.session_state.queue, key=lambda x: ({"High":0, "Medium":1, "Low":2, "Pending":3, "Assigned":4, "In Progress": 5, "Resolved":6}[x['priority']], x['timestamp'])):
            agent_name = ""
            if item["assigned_agent_id"]:
                agent_name = next((a["name"] for a in st.session_state.agents if a["id"] == item["assigned_agent_id"]), "N/A")
            
            priority_display = item['priority']
            if item['suggested_priority_reason']:
                priority_display += f" (AI: {item['suggested_priority_reason']})"

            queue_data.append({
                "ID": item["id"][:8],
                "User": item["user_name"],
                "Issue": item["issue_description"][:50] + "..." if len(item["issue_description"]) > 50 else item["issue_description"],
                "Priority": priority_display,
                "Status": item["status"],
                "Submitted": item["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Agent": agent_name if item["assigned_agent_id"] else "Unassigned"
            })
        st.dataframe(queue_data, use_container_width=True)

        st.subheader("⚖️ Agentic AI: Task Suggestion")
        suggested_task = get_suggested_next_task()
        if suggested_task:
            st.info(f"**Suggested Next Task (by AI):** ID {suggested_task['id'][:8]} - User: {suggested_task['user_name']}, "
                    f"Priority: {suggested_task['priority']}, Issue: {suggested_task['issue_description'][:30]}...")
            
            if st.session_state.agents:
                available_agents = [a for a in st.session_state.agents if a["available"]]
                if available_agents:
                    agent_to_assign_data = st.selectbox("Assign to Agent:", options=[(a["name"], a["id"]) for a in available_agents], format_func=lambda x: x[0], key="dashboard_assign_agent")
                    if agent_to_assign_data: # Check if selection is made
                        agent_to_assign_name, agent_to_assign_id = agent_to_assign_data
                        if st.button(f"Assign Task {suggested_task['id'][:8]} to {agent_to_assign_name}"):
                            if assign_task_to_agent(suggested_task["id"], agent_to_assign_id):
                                st.success(f"Task assigned to {agent_to_assign_name}.")
                                st.rerun()
                            else:
                                st.error("Failed to assign task.")
                else:
                    st.warning("No agents available to assign the task.")
            else:
                st.warning("No agents configured to assign tasks.")
        else:
            st.success("No pending tasks to suggest!")

# --- Tab 3: Agent View ---
with tab3:
    st.header("🧑‍💻 Agent Workspace")
    if not st.session_state.agents:
        st.warning("No agents configured. Please add agents in the sidebar.")
    else:
        selected_agent_id = st.selectbox(
            "Select Agent Profile:",
            options=[a["id"] for a in st.session_state.agents],
            format_func=lambda x: next(a["name"] for a in st.session_state.agents if a["id"] == x)
        )
        
        agent = next((a for a in st.session_state.agents if a["id"] == selected_agent_id), None)

        if agent:
            st.subheader(f"Agent: {agent['name']} (Specialization: {agent['specialization']})")
            st.metric("Status", "Available" if agent["available"] else "Busy")
            st.metric("Tasks Resolved", agent["tasks_resolved"])

            if agent["available"]:
                st.markdown("---")
                st.markdown("#### Take Next Task")
                suggested_task_for_agent = get_suggested_next_task(agent_id=agent["id"])
                if suggested_task_for_agent:
                    st.info(f"**AI Suggests:** Task ID {suggested_task_for_agent['id'][:8]} (Priority: {suggested_task_for_agent['priority']}) for User: {suggested_task_for_agent['user_name']}. Issue: {suggested_task_for_agent['issue_description'][:50]}...")
                    if st.button(f"Take Suggested Task ({suggested_task_for_agent['id'][:8]})", key=f"take_{agent['id']}_{suggested_task_for_agent['id']}"):
                        if assign_task_to_agent(suggested_task_for_agent["id"], agent["id"]):
                            st.success("Task taken!")
                            st.rerun()
                        else:
                            st.error("Could not take task (already assigned or agent became busy).")
                else:
                    st.success("No pending tasks in the queue.")
            
            else: # Agent is busy
                current_task_id = agent["current_task_id"]
                task = next((t for t in st.session_state.queue if t["id"] == current_task_id), None)
                if task:
                    st.markdown("---")
                    st.markdown(f"#### Current Task: ID {task['id'][:8]}")
                    st.markdown(f"**User:** {task['user_name']}")
                    st.markdown(f"**Issue:** {task['issue_description']}")
                    st.markdown(f"**Priority:** {task['priority']}")
                    st.markdown(f"**Submitted:** {task['timestamp'].strftime('%Y-%m-%d %H:%M')}")

                    st.markdown("##### 🤖 AI Assistance (Gemini)")
                    col_gen1, col_gen2 = st.columns(2)
                    with col_gen1:
                        if st.button("✍️ Suggest Response", key=f"suggest_resp_{task['id']}"):
                            if st.session_state.gemini_api_key:
                                prompt = f"A customer '{task['user_name']}' reported the following issue: '{task['issue_description']}'. What is a polite and helpful initial response I can give them? Keep it concise."
                                with st.spinner("Gemini AI is thinking..."):
                                    suggestion = get_gemini_response(prompt) # Switched to Gemini
                                if suggestion:
                                    st.text_area("Suggested Response:", value=suggestion, height=150, key=f"resp_val_{task['id']}")
                            else:
                                st.warning("Google AI API Key needed for suggestions.")
                    with col_gen2:
                        if st.button("📄 Summarize Issue", key=f"summarize_{task['id']}"):
                            if st.session_state.gemini_api_key:
                                prompt = f"Summarize this customer issue in one or two sentences: '{task['issue_description']}'"
                                with st.spinner("Gemini AI is summarizing..."):
                                    summary = get_gemini_response(prompt, max_tokens=100) # Switched to Gemini
                                if summary:
                                    st.info(f"**AI Summary:** {summary}")
                            else:
                                st.warning("Google AI API Key needed for summarization.")
                    
                    st.markdown("---")
                    with st.form(f"resolve_task_form_{task['id']}"):
                        resolution_notes = st.text_area("Resolution Notes:", height=100, key=f"notes_{task['id']}")
                        submitted_resolve = st.form_submit_button("Mark as Resolved")
                        if submitted_resolve:
                            if resolution_notes:
                                if resolve_task(task["id"], resolution_notes):
                                    st.success("Task marked as resolved!")
                                    st.rerun()
                                else:
                                    st.error("Failed to resolve task.")
                            else:
                                st.warning("Please enter resolution notes.")
                else:
                    st.error("Error: Agent is busy but current task not found. This might indicate an inconsistent state.")
                    if st.button("Force Agent to Available (DEBUG)", key=f"force_avail_{agent['id']}"):
                        agent['available'] = True
                        agent['current_task_id'] = None
                        log_event(f"Agent {agent['name']} forced to available status.")
                        st.rerun()

# --- For Debugging: Show Raw Session State ---
# with st.expander("🔍 Show Raw Session State (for debugging)"):
#     st.json(st.session_state.to_dict())