import streamlit as st
import google.generativeai as genai
import uuid
from datetime import datetime, timedelta
import time
import random
import numpy as np # For potential poisson distribution of arrivals

# --- Configuration & Initialization ---
st.set_page_config(layout="wide", page_title="Supermarket Queue Automation")

# Constants
SIMULATION_STEP_MINUTES = 1 # Each step in simulation represents 1 minute
MAX_ITEMS_PER_CUSTOMER = 50
MIN_ITEMS_PER_CUSTOMER = 1
STAFFED_SERVICE_RATE_ITEMS_PER_MIN = 10 # Items a staffed cashier can process per minute
SELF_SERVICE_RATE_ITEMS_PER_MIN = 15  # Items a customer can process at self-service per minute (faster per item but might be overall slower due to user)
CUSTOMER_ARRIVAL_PROBABILITY_PER_STEP = 0.6 # Probability a new customer arrives each minute during peak

# Thresholds for dynamic counter management
OPEN_COUNTER_AVG_WAIT_THRESHOLD_MIN = 5  # If avg wait time > 5 mins, consider opening a counter
CLOSE_COUNTER_AVG_WAIT_THRESHOLD_MIN = 1 # If avg wait time < 1 min and counters idle, consider closing
OPEN_COUNTER_QUEUE_LENGTH_THRESHOLD = 3 # If avg customers per open counter > 3, consider opening

if 'sim_initialized' not in st.session_state:
    st.session_state.gemini_api_key_supermarket = ""
    st.session_state.counters = []
    st.session_state.customers_served_today = [] # To track wait times
    st.session_state.total_customers_in_system = 0
    st.session_state.system_log_supermarket = []
    st.session_state.simulation_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) # Start at 9 AM
    st.session_state.simulation_running = False
    st.session_state.initial_counters_defined = False
    st.session_state.sim_initialized = True


# --- Helper Functions ---
def log_event_supermarket(event_message):
    timestamp = st.session_state.simulation_time.strftime("%H:%M")
    log_entry = f"[{timestamp}] {event_message}"
    st.session_state.system_log_supermarket.append(log_entry)
    if len(st.session_state.system_log_supermarket) > 30:
        st.session_state.system_log_supermarket.pop(0)

def get_gemini_supermarket_response(prompt_text, temperature=0.7, max_tokens=200):
    if not st.session_state.gemini_api_key_supermarket:
        st.error("Google AI API Key not set. Please set it in the sidebar.")
        return None
    try:
        genai.configure(api_key=st.session_state.gemini_api_key_supermarket)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        # Prepend a role for better context (optional, but can help)
        full_prompt = f"You are an AI assistant analyzing a supermarket queue simulation. {prompt_text}"
        response = model.generate_content(full_prompt, generation_config=generation_config)

        if response.parts:
            return response.text
        else:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            st.warning(f"Gemini AI: No content or blocked. Reason: {block_reason}")
            log_event_supermarket(f"Gemini AI: No content or blocked. Reason: {block_reason}")
            return None
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        log_event_supermarket(f"Gemini API Error: {e}")
        return None

def initialize_counters(num_staffed, num_self_service):
    st.session_state.counters = []
    counter_id = 1
    for _ in range(num_staffed):
        st.session_state.counters.append({
            "id": f"S{counter_id}", "type": "Staffed", "status": "Closed", # Start staffed counters closed
            "queue": [], "service_rate": STAFFED_SERVICE_RATE_ITEMS_PER_MIN,
            "current_customer": None, "items_remaining_on_customer": 0,
            "time_busy_current_step": 0
        })
        counter_id += 1
    for _ in range(num_self_service):
        st.session_state.counters.append({
            "id": f"K{counter_id}", "type": "Self-Service", "status": "Open", # Self-service always open
            "queue": [], "service_rate": SELF_SERVICE_RATE_ITEMS_PER_MIN,
            "current_customer": None, "items_remaining_on_customer": 0,
            "time_busy_current_step": 0
        })
        counter_id += 1
    st.session_state.initial_counters_defined = True
    log_event_supermarket(f"Initialized {num_staffed} staffed and {num_self_service} self-service counters.")

def calculate_estimated_wait_time(counter, new_customer_items=0):
    """Calculates estimated time for new customer if they join this counter's queue."""
    wait_time_seconds = 0
    # Time for customers already in queue
    for cust_in_queue in counter["queue"]:
        wait_time_seconds += (cust_in_queue["items"] / counter["service_rate"]) * 60
    # Time for current customer being served
    if counter["current_customer"]:
        wait_time_seconds += (counter["items_remaining_on_customer"] / counter["service_rate"]) * 60
    # Time for the new customer
    if new_customer_items > 0:
         wait_time_seconds += (new_customer_items / counter["service_rate"]) * 60
    return wait_time_seconds / 60 # in minutes

def assign_customer_to_optimal_counter(customer):
    """Agentic AI: Assigns customer to the best available open counter."""
    best_counter = None
    min_wait_time = float('inf')

    open_counters = [c for c in st.session_state.counters if c["status"] == "Open"]
    if not open_counters:
        log_event_supermarket(f"No open counters! Customer {customer['id'][:4]} has to wait globally.")
        # In a more complex system, this customer would go to a "waiting area"
        # For simplicity, we'll assume this scenario implies a need to open a counter.
        return False # Indicate assignment failed

    for counter in open_counters:
        est_wait = calculate_estimated_wait_time(counter, customer["items"])
        # Prefer self-service for small baskets if available and wait times are comparable
        if customer["items"] <= 10 and counter["type"] == "Self-Service" and est_wait < min_wait_time + 0.5: # Add slight preference
            min_wait_time = est_wait
            best_counter = counter
        elif est_wait < min_wait_time:
            min_wait_time = est_wait
            best_counter = counter

    if best_counter:
        best_counter["queue"].append(customer)
        log_event_supermarket(f"Customer {customer['id'][:4]} ({customer['items']} items) assigned to {best_counter['type']} Counter {best_counter['id']}. Est. wait: {min_wait_time:.1f} min.")
        return True
    else: # Should not happen if open_counters is not empty, but as a fallback
        log_event_supermarket(f"Could not find an optimal counter for Customer {customer['id'][:4]}.")
        return False


def manage_counter_availability_agent():
    """Agentic AI: Opens/Closes staffed counters based on demand."""
    open_staffed_counters = [c for c in st.session_state.counters if c["type"] == "Staffed" and c["status"] == "Open"]
    closed_staffed_counters = [c for c in st.session_state.counters if c["type"] == "Staffed" and c["status"] == "Closed"]
    
    all_open_counters = [c for c in st.session_state.counters if c["status"] == "Open"]
    if not all_open_counters: # Ensure at least one self-service or ability to open a staffed one
        if closed_staffed_counters:
            counter_to_open = closed_staffed_counters[0]
            counter_to_open["status"] = "Open"
            log_event_supermarket(f"AGENT DECISION: No counters open! Opening Staffed Counter {counter_to_open['id']}.")
            return "Opened a counter as none were active."
        return "No counters to manage."


    total_customers_waiting = sum(len(c["queue"]) for c in all_open_counters) + sum(1 for c in all_open_counters if c["current_customer"])
    avg_customers_per_open_counter = total_customers_waiting / len(all_open_counters) if all_open_counters else 0
    
    # Calculate current average wait time (simplified: for customers at head of queue)
    current_wait_times = []
    for c in all_open_counters:
        if c["queue"]: # consider only those with queues for this metric
             # time for current customer + first in queue
            base_wait = (c["items_remaining_on_customer"] / c["service_rate"]) * 60 if c["current_customer"] else 0
            current_wait_times.append((base_wait + (c["queue"][0]["items"] / c["service_rate"]) * 60) / 60)

    avg_wait_time_system = np.mean(current_wait_times) if current_wait_times else 0

    decision_reason = ""

    # Rule to OPEN a staffed counter
    if closed_staffed_counters:
        if avg_wait_time_system > OPEN_COUNTER_AVG_WAIT_THRESHOLD_MIN or avg_customers_per_open_counter > OPEN_COUNTER_QUEUE_LENGTH_THRESHOLD :
            counter_to_open = closed_staffed_counters[0] # Open the first available closed one
            counter_to_open["status"] = "Open"
            decision_reason = f"Avg wait time {avg_wait_time_system:.1f} min (>{OPEN_COUNTER_AVG_WAIT_THRESHOLD_MIN}) and/or avg queue {avg_customers_per_open_counter:.1f} (>{OPEN_COUNTER_QUEUE_LENGTH_THRESHOLD}). Opening Staffed Counter {counter_to_open['id']}."
            log_event_supermarket(f"AGENT DECISION: {decision_reason}")
            return decision_reason

    # Rule to CLOSE a staffed counter (if more than one is open)
    if len(open_staffed_counters) > 1: # Keep at least one staffed counter open if it's the only type or as a policy
        if avg_wait_time_system < CLOSE_COUNTER_AVG_WAIT_THRESHOLD_MIN and avg_customers_per_open_counter < (OPEN_COUNTER_QUEUE_LENGTH_THRESHOLD / 2):
            # Find an idle staffed counter to close
            for counter in open_staffed_counters:
                if not counter["current_customer"] and not counter["queue"]:
                    counter["status"] = "Closed"
                    decision_reason = f"Low demand: Avg wait time {avg_wait_time_system:.1f} min (<{CLOSE_COUNTER_AVG_WAIT_THRESHOLD_MIN}) & avg queue {avg_customers_per_open_counter:.1f}. Closing Staffed Counter {counter['id']}."
                    log_event_supermarket(f"AGENT DECISION: {decision_reason}")
                    return decision_reason
    return "No change in counter status needed based on current load."


def process_counters_step():
    """Simulates one time step of processing at all counters."""
    for counter in st.session_state.counters:
        if counter["status"] != "Open":
            counter["time_busy_current_step"] = 0
            continue

        counter["time_busy_current_step"] = 0 # Reset for this step
        if counter["current_customer"]:
            items_processed_this_step = counter["service_rate"] * (SIMULATION_STEP_MINUTES)
            counter["items_remaining_on_customer"] -= items_processed_this_step
            counter["time_busy_current_step"] = SIMULATION_STEP_MINUTES # Assume busy for the whole step if processing

            if counter["items_remaining_on_customer"] <= 0:
                served_customer = counter["current_customer"]
                wait_time = (st.session_state.simulation_time - served_customer["arrival_time"]).total_seconds() / 60
                st.session_state.customers_served_today.append({"id": served_customer["id"], "wait_time_min": wait_time, "items": served_customer["items"]})
                log_event_supermarket(f"Customer {served_customer['id'][:4]} ({served_customer['items']} items) served at Counter {counter['id']}. Wait: {wait_time:.1f} min.")
                counter["current_customer"] = None
                counter["items_remaining_on_customer"] = 0
                st.session_state.total_customers_in_system -=1


        # If counter is idle and has a queue, pick next customer
        if not counter["current_customer"] and counter["queue"]:
            next_customer = counter["queue"].pop(0)
            counter["current_customer"] = next_customer
            counter["items_remaining_on_customer"] = next_customer["items"]
            # time_busy_current_step will be counted in the next step when items are processed
            log_event_supermarket(f"Counter {counter['id']} started serving Customer {next_customer['id'][:4]} ({next_customer['items']} items).")


def simulate_one_step():
    st.session_state.simulation_time += timedelta(minutes=SIMULATION_STEP_MINUTES)
    log_event_supermarket(f"--- Simulation Step: Time is now {st.session_state.simulation_time.strftime('%H:%M')} ---")

    # 1. Manage counter availability (Agentic AI)
    decision_log = manage_counter_availability_agent()
    if decision_log and "No change" not in decision_log and "No counters to manage" not in decision_log:
        st.session_state.last_counter_management_decision = decision_log


    # 2. New customer arrivals
    # More customers during peak hours (e.g. 12-2 PM, 5-7 PM)
    current_hour = st.session_state.simulation_time.hour
    arrival_prob = CUSTOMER_ARRIVAL_PROBABILITY_PER_STEP
    if 12 <= current_hour < 14 or 17 <= current_hour < 19:
        arrival_prob *= 1.5 # Higher arrival rate during peak
    if random.random() < arrival_prob:
        num_items = random.randint(MIN_ITEMS_PER_CUSTOMER, MAX_ITEMS_PER_CUSTOMER)
        customer_id = str(uuid.uuid4())
        new_customer = {"id": customer_id, "items": num_items, "arrival_time": st.session_state.simulation_time}
        st.session_state.total_customers_in_system +=1
        log_event_supermarket(f"New customer {customer_id[:4]} arrived with {num_items} items.")
        assign_customer_to_optimal_counter(new_customer)

    # 3. Process items at each counter
    process_counters_step()

    # 4. Update overall metrics (could be done less frequently for performance)
    # (Metrics like avg wait time are better calculated when customers complete service)


# --- Streamlit UI ---

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ System Setup & Controls")
    st.session_state.gemini_api_key_supermarket = st.text_input(
        "Google AI Studio API Key",
        type="password",
        value=st.session_state.gemini_api_key_supermarket,
        help="Get your key from Google AI Studio (makersuite.google.com)",
        key="gemini_api_key_sb"
    )

    if not st.session_state.initial_counters_defined:
        st.subheader("Define Counters")
        num_staffed_init = st.number_input("Number of Staffed Counters", min_value=0, max_value=10, value=3, key="num_staffed_sb")
        num_self_service_init = st.number_input("Number of Self-Service Kiosks", min_value=0, max_value=10, value=4, key="num_self_sb")
        if st.button("Initialize Supermarket Layout"):
            if num_staffed_init + num_self_service_init > 0:
                initialize_counters(num_staffed_init, num_self_service_init)
                st.success("Supermarket layout initialized!")
                st.rerun()
            else:
                st.error("Please define at least one counter.")
    else:
        st.success("Counters Initialized.")
        if st.button("Reset Simulation and Counters"):
            st.session_state.sim_initialized = False # Trigger re-init block
            st.session_state.initial_counters_defined = False
            # Clear other relevant states
            st.session_state.customers_served_today = []
            st.session_state.total_customers_in_system = 0
            st.session_state.system_log_supermarket = []
            st.session_state.simulation_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
            st.session_state.simulation_running = False
            st.session_state.last_counter_management_decision = ""
            st.session_state.gemini_insights = ""
            st.rerun()


    if st.session_state.initial_counters_defined:
        st.subheader("Simulation Control")
        if st.session_state.simulation_running:
            if st.button("⏹️ Pause Simulation", key="pause_sim_sb"):
                st.session_state.simulation_running = False
                log_event_supermarket("Simulation Paused.")
                st.rerun()
        else:
            if st.button("▶️ Run Simulation", key="run_sim_sb"):
                st.session_state.simulation_running = True
                log_event_supermarket("Simulation Started/Resumed.")
                st.rerun()

        if not st.session_state.simulation_running:
             if st.button("⏭️ Simulate Next Step", key="next_step_sb"):
                simulate_one_step()
                st.rerun()

        simulation_speed = st.slider("Simulation Speed (steps per second when running)", 0.1, 5.0, 1.0, 0.1, key="sim_speed_sb", disabled=not st.session_state.simulation_running)

    st.subheader("📜 System Log")
    log_container_sb = st.container(height=250)
    for log in reversed(st.session_state.system_log_supermarket):
        log_container_sb.caption(log)

# Main Application Area
st.title("🛒 Supermarket Intelligent Queue Automation")
st.markdown(f"**Simulation Time: {st.session_state.simulation_time.strftime('%A, %B %d, %Y %H:%M')}**")

if not st.session_state.initial_counters_defined:
    st.info("Please initialize the supermarket layout from the sidebar.")
else:
    # Display Counters
    st.header("🏪 Counter Status")
    cols = st.columns(len(st.session_state.counters) if st.session_state.counters else 1)
    for i, counter in enumerate(st.session_state.counters):
        with cols[i % len(cols)]: # Cycle through columns
            status_emoji = "✅" if counter["status"] == "Open" else "❌"
            busy_emoji = "🧑‍💻" if counter["current_customer"] else ("⏳" if counter["queue"] else "🧘")
            
            card_color = "lightgreen" if counter["status"] == "Open" else "lightcoral"
            if counter["status"] == "Open" and counter["current_customer"]:
                card_color = "lightblue"
            elif counter["status"] == "Open" and counter["queue"]:
                card_color = "lightyellow"


            st.markdown(f"""
            <div style="background-color:{card_color}; padding:10px; border-radius:5px; margin-bottom:10px;">
                <h4>{counter['type']} {counter['id']} {status_emoji} {busy_emoji}</h4>
                <b>Status:</b> {counter['status']}<br>
                <b>Queue Length:</b> {len(counter['queue'])}<br>
                <b>Serving:</b> {counter['current_customer']['id'][:4] if counter['current_customer'] else 'None'} ({counter['items_remaining_on_customer']:.0f} items left)<br>
                <b>Est. Wait (next):</b> {calculate_estimated_wait_time(counter):.1f} min
            </div>
            """, unsafe_allow_html=True)
            # Display queue details
            if counter["queue"]:
                with st.expander(f"View Queue ({len(counter['queue'])} customers)"):
                    for cust_idx, cust in enumerate(counter["queue"]):
                        st.caption(f"{cust_idx+1}. Cust {cust['id'][:4]} ({cust['items']} items)")


    # Metrics Display
    st.header("📊 System Metrics")
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    total_served = len(st.session_state.customers_served_today)
    avg_wait_overall = np.mean([c["wait_time_min"] for c in st.session_state.customers_served_today]) if total_served > 0 else 0
    
    col_metric1.metric("Total Customers in System Now", st.session_state.total_customers_in_system)
    col_metric2.metric("Total Customers Served Today", total_served)
    col_metric3.metric("Avg. Wait Time (Served)", f"{avg_wait_overall:.1f} min")

    open_counters_now = [c for c in st.session_state.counters if c["status"]=="Open"]
    utilization_current_step = 0
    if open_counters_now:
        busy_time_this_step = sum(c.get("time_busy_current_step", 0) for c in open_counters_now)
        total_possible_time_this_step = len(open_counters_now) * SIMULATION_STEP_MINUTES
        utilization_current_step = (busy_time_this_step / total_possible_time_this_step) * 100 if total_possible_time_this_step > 0 else 0
    
    st.metric("Overall Counter Utilization (Current Step)", f"{utilization_current_step:.1f}%")


    # Generative AI Insights
    st.header("💡 AI Insights & Suggestions (Gemini)")
    if 'last_counter_management_decision' in st.session_state and st.session_state.last_counter_management_decision:
        st.info(f"**Last Agent Decision:** {st.session_state.last_counter_management_decision}")

    if st.button("🤖 Get Gemini Analysis & Suggestions", key="gemini_analysis_sb"):
        if st.session_state.gemini_api_key_supermarket:
            # Create a snapshot of the current state for Gemini
            current_status_prompt = "Current supermarket state:\n"
            current_status_prompt += f"- Simulation Time: {st.session_state.simulation_time.strftime('%H:%M')}\n"
            current_status_prompt += f"- Total customers in system: {st.session_state.total_customers_in_system}\n"
            current_status_prompt += f"- Average wait time for served customers: {avg_wait_overall:.1f} minutes\n"
            current_status_prompt += "- Counter States:\n"
            for c in st.session_state.counters:
                current_status_prompt += f"  - {c['type']} {c['id']}: Status={c['status']}, Queue={len(c['queue'])}, Serving={'Yes' if c['current_customer'] else 'No'}\n"
            if 'last_counter_management_decision' in st.session_state and st.session_state.last_counter_management_decision:
                 current_status_prompt += f"- Last automated decision: {st.session_state.last_counter_management_decision}\n"

            prompt_for_gemini = f"{current_status_prompt}\nBased on this, provide a brief (2-3 sentences) summary of the current situation and one actionable suggestion to improve queue flow or customer experience. If an automated decision was just made, briefly explain its rationale if it's not obvious."

            with st.spinner("Gemini is analyzing the situation..."):
                st.session_state.gemini_insights = get_gemini_supermarket_response(prompt_for_gemini, temperature=0.5)
            if st.session_state.gemini_insights:
                st.success("Analysis Complete!")
            else:
                st.error("Could not get analysis from Gemini.")
        else:
            st.warning("Please enter your Google AI API Key in the sidebar.")

    if 'gemini_insights' in st.session_state and st.session_state.gemini_insights:
        st.markdown("**Gemini's Analysis:**")
        st.markdown(f"> {st.session_state.gemini_insights}")


# Auto-run simulation if toggled
if st.session_state.get('simulation_running', False) and st.session_state.initial_counters_defined:
    simulate_one_step()
    time.sleep(1.0 / simulation_speed) # Control speed
    st.rerun()
