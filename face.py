import streamlit as st
import datetime
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Define Google LLM for interacting with Google Calendar
llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.6, google_api_key="AIzaSyDjITo6JpwACzQKlMCJKuBhHHK8jTQIhBg") #google api key

# Define Farmer Agent
farmer_agent = Agent(
    role='Farmer Agent',
    goal='Gather planting information from the farmer',
    backstory='An agent specialized in interacting with farmers to gather planting information.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Agronomist Agent
agronomist_agent = Agent(
    role='Agronomist Local Expert at this city',
    goal='Provide best personalized farming advice based on weather, season, and prices of the selected city',
    backstory='An expert that specialized in providing personalized farming advice based on location and crop.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Planner Agent
planner_agent = Agent(
    role='Amazing Planner Agent',
    goal='Create the most amazing planting calendar with budget and best farming practice ',
    backstory='Specialist in farm management an agronomist with decades of experience calendar based on the provided information.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Crop Suggestion Agent
crop_suggestion_agent = Agent(
    role='Crop Suggestion Agent',
    goal='Suggest alternative crops if the entered crop is out of season',
    backstory='An agent specialized in suggesting alternative crops based on seasonality and profitability.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Task for gathering planting information from the farmer
planting_info_task = Task(
    description='Gather planting information from the farmer: {plant}',
    agent=farmer_agent,
    expected_output='Planting information collected from the farmer.'
)

# Define Task for providing personalized farming advice
farming_advice_task = Task(
    description='Provide personalized farming advice for {crop} in {location} starting from {start_date}.',
    agent=agronomist_agent,
    expected_output='Personalized farming advice provided.'
)

# Define Task for generating farming calendar
farming_calendar_task = Task(
    description='Generate farming calendar for {crop} in {location} starting from {start_date}.',
    agent=planner_agent,
    expected_output='Farming calendar generated.'
)

# Define Task for advising if planting season has ended
season_check_task = Task(
    description='Check if the planting season has ended for {crop} in {location} by {current_date}.',
    agent=agronomist_agent,
    expected_output='Planting season status checked.'
)

# Define Task for suggesting alternative crops
crop_suggestion_task = Task(
    description='Suggest alternative crops if {crop} is out of season for {location} by {current_date}.',
    agent=crop_suggestion_agent,
    expected_output='Alternative crops suggested.'
)

# Define Task for displaying farming itinerary
farming_itinerary_task = Task(
    description='Display farming itinerary for {crop} in {location} starting from {start_date}.',
    agent=agronomist_agent,
    expected_output='Farming itinerary displayed.'
)

# Create a Crew for managing the farming process
farming_crew = Crew(
    agents=[farmer_agent, agronomist_agent, planner_agent, crop_suggestion_agent],
    tasks=[planting_info_task, farming_advice_task, farming_calendar_task, season_check_task, crop_suggestion_task, farming_itinerary_task],
    verbose=True,
    process=Process.sequential
)

# Streamlit App
st.title("AbutiSpinach: Your Farming Assistant")

# Welcome message
st.write("Welcome to AbutiSpinach! Your go-to assistant for all your farming needs.")

# Gather planting information from the farmer
location = st.text_input("Enter your location:")
crop = st.text_input("Enter the crop you want to plant:")
start_date = st.date_input("Enter the date you want to start planting:")

if st.button("Submit"):
    if not location or not crop or not start_date:
        st.error("Please fill out all fields.")
    else:
        try:
            # Interpolate farmer's planting information into the tasks descriptions
            planting_info_task.interpolate_inputs({"plant": crop})
            farming_advice_task.interpolate_inputs({"crop": crop, "location": location, "start_date": start_date})
            farming_calendar_task.interpolate_inputs({"crop": crop, "location": location, "start_date": start_date})
            current_date = datetime.date.today()
            season_check_task.interpolate_inputs({"crop": crop, "location": location, "current_date": current_date})
            crop_suggestion_task.interpolate_inputs({"crop": crop, "location": location, "current_date": current_date})

            # Execute the farming crew
            st.write("Executing farming tasks...")

            # Execute each task sequentially
            for task in farming_crew.tasks:
                st.write(f"Executing task: {task.description}")
                task.execute()
                st.success("Task completed successfully!")
            
            # Display farming itinerary
            farming_itinerary = farming_itinerary_task.output
            st.subheader("Farming Itinerary:")
            st.write(farming_itinerary)  # Display farming itinerary as plain text
        except ValueError:
            st.error("Invalid input. Please enter valid values.")

