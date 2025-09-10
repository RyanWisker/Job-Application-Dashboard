import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import pickle
import hashlib

# Page configuration
st.set_page_config(
    page_title="Summer 2025 Job Application Tracker",
    page_icon="📊",
    layout="wide"
)

# Data persistence configuration
DATA_DIR = "job_tracker_data"
APPLICATIONS_FILE = os.path.join(DATA_DIR, "applications.json")
RESUME_FILE = os.path.join(DATA_DIR, "resume.txt")
BACKUP_FILE = os.path.join(DATA_DIR, "applications_backup.json")

def ensure_data_directory():
    """Ensure data directory exists"""
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
        except:
            pass

def get_user_session_id():
    """Generate a unique session ID based on browser session"""
    # Use Streamlit's session state to maintain consistency
    if 'user_session_id' not in st.session_state:
        # Create a simple session ID (in production, you might want user authentication)
        import time
        st.session_state.user_session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return st.session_state.user_session_id

def get_user_data_files():
    """Get user-specific data file paths"""
    session_id = get_user_session_id()
    return {
        'applications': os.path.join(DATA_DIR, f"applications_{session_id}.json"),
        'resume': os.path.join(DATA_DIR, f"resume_{session_id}.txt"),
        'backup': os.path.join(DATA_DIR, f"backup_{session_id}.json")
    }

def save_applications_data():
    """Save applications to JSON file with backup"""
    ensure_data_directory()
    files = get_user_data_files()
    
    try:
        # Create backup of existing data
        if os.path.exists(files['applications']):
            try:
                with open(files['applications'], 'r') as f:
                    backup_data = json.load(f)
                with open(files['backup'], 'w') as f:
                    json.dump(backup_data, f, indent=2)
            except:
                pass
        
        # Save current data
        with open(files['applications'], 'w') as f:
            json.dump(st.session_state.applications, f, indent=2, default=str)
        
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def load_applications_data():
    """Load applications from JSON file"""
    ensure_data_directory()
    files = get_user_data_files()
    
    try:
        if os.path.exists(files['applications']):
            with open(files['applications'], 'r') as f:
                data = json.load(f)
                st.session_state.applications = data
                return True
        else:
            st.session_state.applications = []
            return False
    except Exception as e:
        st.error(f"Error loading applications data: {str(e)}")
        # Try to load from backup
        try:
            if os.path.exists(files['backup']):
                with open(files['backup'], 'r') as f:
                    data = json.load(f)
                    st.session_state.applications = data
                    st.warning("Loaded data from backup file.")
                    return True
        except:
            pass
        
        st.session_state.applications = []
        return False

def save_resume_data():
    """Save resume to text file"""
    ensure_data_directory()
    files = get_user_data_files()
    
    try:
        with open(files['resume'], 'w', encoding='utf-8') as f:
            f.write(st.session_state.get('resume', ''))
        return True
    except Exception as e:
        st.error(f"Error saving resume: {str(e)}")
        return False

def load_resume_data():
    """Load resume from text file"""
    ensure_data_directory()
    files = get_user_data_files()
    
    try:
        if os.path.exists(files['resume']):
            with open(files['resume'], 'r', encoding='utf-8') as f:
                st.session_state.resume = f.read()
                return True
        else:
            st.session_state.resume = ""
            return False
    except Exception as e:
        st.error(f"Error loading resume: {str(e)}")
        st.session_state.resume = ""
        return False

def initialize_session_state():
    """Initialize all session state variables"""
    if 'applications' not in st.session_state:
        st.session_state.applications = []
    
    if 'resume' not in st.session_state:
        st.session_state.resume = ""
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def load_all_data():
    """Load all user data"""
    if not st.session_state.data_loaded:
        load_applications_data()
        load_resume_data()
        st.session_state.data_loaded = True

def export_all_data():
    """Export all data as JSON for backup/transfer"""
    export_data = {
        'applications': st.session_state.applications,
        'resume': st.session_state.resume,
        'export_date': datetime.now().isoformat(),
        'session_id': get_user_session_id()
    }
    return json.dumps(export_data, indent=2, default=str)

def import_data(imported_json):
    """Import data from JSON"""
    try:
        data = json.loads(imported_json)
        
        if 'applications' in data:
            st.session_state.applications = data['applications']
            save_applications_data()
        
        if 'resume' in data:
            st.session_state.resume = data['resume']
            save_resume_data()
        
        return True
    except Exception as e:
        st.error(f"Error importing data: {str(e)}")
        return False

# Status options and colors
STATUS_OPTIONS = [
    "Not Applied", "Applied", "Interview Scheduled", 
    "Interviewed", "Offer Received", "Rejected", "Withdrawn"
]

STATUS_COLORS = {
    "Not Applied": "#9CA3AF",
    "Applied": "#3B82F6", 
    "Interview Scheduled": "#F59E0B",
    "Interviewed": "#8B5CF6",
    "Offer Received": "#10B981",
    "Rejected": "#EF4444",
    "Withdrawn": "#6B7280"
}

SOURCE_OPTIONS = ["LinkedIn", "Handshake", "Company Website", "Indeed", "Referral", "Other"]

def add_application(job_title, company_name, location, wage, status, source, job_description):
    """Add new application to the list"""
    # Generate unique ID based on existing applications
    existing_ids = [app.get('id', 0) for app in st.session_state.applications]
    new_id = max(existing_ids, default=0) + 1
    
    application = {
        'id': new_id,
        'job_title': job_title,
        'company_name': company_name,
        'location': location,
        'wage': wage,
        'status': status,
        'source': source,
        'job_description': job_description,
        'date_added': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    st.session_state.applications.append(application)
    save_applications_data()

def delete_application(app_id):
    """Delete application by ID"""
    st.session_state.applications = [app for app in st.session_state.applications if app['id'] != app_id]
    save_applications_data()

def update_application_status(app_id, new_status):
    """Update application status"""
    for app in st.session_state.applications:
        if app['id'] == app_id:
            app['status'] = new_status
            break
    save_applications_data()

def update_application(app_id, **kwargs):
    """Update application with any field"""
    for app in st.session_state.applications:
        if app['id'] == app_id:
            for key, value in kwargs.items():
                if key in app:
                    app[key] = value
            break
    save_applications_data()


def generate_cover_letter(application):
    """Generate tailored cover letter using OpenAI GPT-4o Mini"""
    openai_key = st.session_state.get('openai_api_key', '').strip()
    
    if not openai_key:
        return "❌ Please enter your OpenAI API key in the AI Settings page first."
    
    if not st.session_state.resume.strip():
        return "❌ Please add your resume information first in the Resume section."
    
    # Combine resume and job description into a single prompt
    prompt = f"""
You are a professional resume and cover letter writer.
Write a **one-page, professional cover letter** that is tailored to the following job description and resume information.
Make it concise, engaging, and highlight the applicant's strengths. Also, do not use the word "fosetering" and no hyphens.

--- Job Description ---
{application['job_description']}

--- Resume Information ---
{st.session_state.resume}

Include the applicant's name as [Your Name] and the company as {application['company_name']}.
Keep the tone professional and enthusiastic.
    """
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        
        # Correct way to get the content
        cover_letter = response.choices[0].message.content.strip()
        return cover_letter
    
    except Exception as e:
        return f"❌ API call failed: {str(e)}"


def create_sankey_chart():
    """Create Sankey-style visualization using plotly"""
    if not st.session_state.applications:
        return None
    
    df = pd.DataFrame(st.session_state.applications)
    status_counts = df['status'].value_counts()
    
    # Create horizontal bar chart that looks like a Sankey
    fig = go.Figure()
    
    for i, (status, count) in enumerate(status_counts.items()):
        fig.add_trace(go.Bar(
            y=[status],
            x=[count],
            name=status,
            orientation='h',
            marker_color=STATUS_COLORS.get(status, '#9CA3AF'),
            text=[f"{count} ({count/len(df)*100:.1f}%)"],
            textposition='inside'
        ))
    
    fig.update_layout(
        title="Application Process Flow",
        xaxis_title="Number of Applications",
        yaxis_title="Status",
        showlegend=False,
        height=400
    )
    
    return fig

def analytics_dashboard():
    """Create analytics dashboard"""
    if not st.session_state.applications:
        st.warning("No applications to analyze. Add some applications first!")
        return
    
    df = pd.DataFrame(st.session_state.applications)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_apps = len(df)
    applied_count = len(df[df['status'].isin(['Applied', 'Interview Scheduled', 'Interviewed', 'Offer Received', 'Rejected'])])
    interview_count = len(df[df['status'].isin(['Interview Scheduled', 'Interviewed', 'Offer Received'])])
    offer_count = len(df[df['status'] == 'Offer Received'])
    
    with col1:
        st.metric("Total Applications", total_apps)
    
    with col2:
        st.metric("Actually Applied", applied_count)
    
    with col3:
        response_rate = (interview_count / applied_count * 100) if applied_count > 0 else 0
        st.metric("Response Rate", f"{response_rate:.1f}%")
    
    with col4:
        st.metric("Offers Received", offer_count)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_counts = df['status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Application Status Distribution",
            color=status_counts.index,
            color_discrete_map=STATUS_COLORS
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Source distribution
        source_counts = df['source'].value_counts()
        fig_source = px.bar(
            x=source_counts.index,
            y=source_counts.values,
            title="Applications by Source",
            labels={'x': 'Source', 'y': 'Count'}
        )
        st.plotly_chart(fig_source, use_container_width=True)
    
    # Sankey-style flow chart
    sankey_fig = create_sankey_chart()
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)

def data_management_page():
    """Data management and backup page"""
    st.header("💾 Data Management")
    
    # Session info
    st.subheader("📊 Session Information")
    session_id = get_user_session_id()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Session ID", session_id)
    with col2:
        st.metric("Applications", len(st.session_state.applications))
    with col3:
        resume_chars = len(st.session_state.resume)
        st.metric("Resume Characters", resume_chars)
    
    st.divider()
    
    # Data backup and export
    st.subheader("📤 Export Data")
    st.write("Download your data as a backup or to transfer to another device.")
    
    if st.button("📋 Generate Export Data"):
        export_json = export_all_data()
        st.session_state.export_data = export_json
    
    if 'export_data' in st.session_state:
        st.download_button(
            label="⬇️ Download Data Backup",
            data=st.session_state.export_data,
            file_name=f"job_tracker_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    st.divider()
    
    # Data import
    st.subheader("📤 Import Data")
    st.write("Upload a previously exported backup file to restore your data.")
    
    uploaded_file = st.file_uploader("Choose backup file", type=['json'])
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode('utf-8')
            if st.button("🔄 Import Data"):
                if import_data(file_content):
                    st.success("Data imported successfully!")
                    st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.divider()
    
    # Manual data input
    st.subheader("✏️ Manual Data Entry")
    with st.expander("Import from JSON text"):
        json_text = st.text_area("Paste JSON data here:", height=200)
        if st.button("Import from Text"):
            if json_text.strip():
                if import_data(json_text):
                    st.success("Data imported from text successfully!")
                    st.rerun()
    
    st.divider()
    
    # Clear data
    st.subheader("🗑️ Clear Data")
    st.warning("This will permanently delete all your applications and resume data!")
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            st.session_state.applications = []
            st.session_state.resume = ""
            save_applications_data()
            save_resume_data()
            st.success("All data cleared!")
            st.session_state.confirm_clear = False
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.error("Click again to confirm deletion!")

def main():
    # Initialize session state and load data
    initialize_session_state()
    load_all_data()
    
    st.title("📊 Summer 2025 Job Application Tracker")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "Application Tracker", 
        "Analytics", 
        "Resume", 
        "Data Management",
        "AI Settings"
    ])
    
    if page == "Data Management":
        data_management_page()
        
    elif page == "AI Settings":
        st.header("🤖 AI Settings")
        st.write("Configure your AI API keys for enhanced cover letter generation.")
        
        # API Key inputs
        st.subheader("🔧 OpenAI GPT-4o Mini Configuration")
        st.info("GPT-4o Mini is OpenAI's most cost-effective model, perfect for cover letters at ~$0.001 per letter!")
        
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.get('openai_api_key', ''),
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
            st.success("✅ OpenAI API Key saved!")
        
        # Test API connection
        if st.button("🧪 Test API Connection"):
            if openai_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    
                    # Simple test call
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "Say 'API connection successful!'"}],
                        max_tokens=10
                    )
                    st.success("✅ API connection successful! Ready to generate cover letters.")
                except Exception as e:
                    st.error(f"❌ API connection failed: {str(e)}")
            else:
                st.error("Please enter your API key first.")
        
        # Cost calculator
        st.subheader("💰 Cost Calculator")
        num_letters = st.slider("Number of cover letters to generate:", 1, 100, 10)
        estimated_cost = num_letters * 0.001  # Approximate cost per letter
        st.metric("Estimated Cost", f"${estimated_cost:.3f}")
        st.caption("GPT-4o Mini is extremely cost-effective - about 80% cheaper than GPT-4!")
        
        # Model comparison focused on GPT-4o Mini
        st.subheader("🏆 Why GPT-4o Mini for Cover Letters?")
        
        benefits = {
            "Feature": [
                "Cost per Cover Letter",
                "Quality", 
                "Speed",
                "Context Understanding",
                "Professional Writing",
                "Customization"
            ],
            "GPT-4o Mini": [
                "~$0.001",
                "Excellent",
                "Very Fast",
                "Superior",
                "Professional",
                "Highly Tailored"
            ],
            "GPT-3.5 Turbo": [
                "~$0.002",
                "Good",
                "Fast",
                "Good",
                "Good",
                "Moderate"
            ],
            "Free Alternatives": [
                "$0",
                "Basic",
                "Slow",
                "Limited",
                "Generic",
                "Minimal"
            ]
        }
        
        st.table(pd.DataFrame(benefits))
        
        # Instructions
        st.subheader("🚀 Setup Instructions")
        
        with st.expander("OpenAI Setup"):
            st.markdown("""
            1. Go to https://platform.openai.com/api-keys
            2. Create an account and add billing information
            3. Create a new API key
            4. Copy and paste it above
            5. Cost: ~$0.002 per cover letter with GPT-3.5-turbo
            """)
        
        with st.expander("Anthropic Setup"):
            st.markdown("""
            1. Go to https://console.anthropic.com/
            2. Create an account
            3. Generate an API key
            4. Copy and paste it above
            5. Cost: Similar to OpenAI
            """)
        
        with st.expander("Hugging Face Setup (FREE)"):
            st.markdown("""
            1. Go to https://huggingface.co/settings/tokens
            2. Create a free account
            3. Generate a new token
            4. Copy and paste it above
            5. Cost: Free with rate limits
            """)
        
        with st.expander("Local LLM Setup (Advanced)"):
            st.markdown("""
            1. Install Ollama: https://ollama.ai/
            2. Run: `ollama pull llama2`
            3. Start Ollama server
            4. Select "local" in the AI model dropdown
            5. Cost: Free, runs on your computer
            """)
    
    elif page == "Application Tracker":
        st.header("🎯 Application Tracker")
        
        # Add new application form
        with st.expander("➕ Add New Application", expanded=False):
            with st.form("add_application"):
                col1, col2 = st.columns(2)
                
                with col1:
                    job_title = st.text_input("Job Title *", placeholder="e.g., Software Engineer Intern")
                    location = st.text_input("Location *", placeholder="e.g., New York, NY or Remote")
                    status = st.selectbox("Status", STATUS_OPTIONS, index=0)
                
                with col2:
                    company_name = st.text_input("Company Name *", placeholder="e.g., Google")
                    wage = st.text_input("Wage/Salary *", placeholder="e.g., $70,000 or $25/hour")
                    source = st.selectbox("Source", SOURCE_OPTIONS, index=0)
                
                job_description = st.text_area(
                    "Job Description *", 
                    placeholder="Paste the complete job description here...",
                    height=150
                )
                
                submitted = st.form_submit_button("Add Application")
                
                if submitted:
                    if job_title and company_name and location and wage and job_description:
                        add_application(job_title, company_name, location, wage, status, source, job_description)
                        st.success("Application added successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields marked with *")
        
        # Display applications
        if st.session_state.applications:
            st.subheader("📋 Your Applications")
            
            # Sort applications by date (newest first)
            sorted_apps = sorted(st.session_state.applications, 
                                key=lambda x: x.get('date_added', ''), reverse=True)
            
            for i, app in enumerate(sorted_apps):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    
                    with col1:
                        st.write(f"**{app['job_title']}** at **{app['company_name']}**")
                        st.write(f"📍 {app['location']} | 💰 {app['wage']} | 📅 {app['date_added']}")
                    
                    with col2:
                        # Status badge
                        status_color = STATUS_COLORS.get(app['status'], '#9CA3AF')
                        st.markdown(f"""
                        <span style="background-color: {status_color}; color: white; padding: 4px 8px; 
                        border-radius: 12px; font-size: 12px; font-weight: bold;">
                        {app['status']}
                        </span>
                        """, unsafe_allow_html=True)
                        st.write(f"Source: {app['source']}")
                    
                    with col3:
                        # Generate cover letter button
                        if st.button("📄 Cover Letter", key=f"cover_{app['id']}"):
                            cover_letter = generate_cover_letter(app)
                            st.session_state[f'cover_letter_{app["id"]}'] = cover_letter
                    
                    with col4:
                        # Update status
                        new_status = st.selectbox(
                            "Update Status",
                            STATUS_OPTIONS,
                            index=STATUS_OPTIONS.index(app['status']),
                            key=f"status_{app['id']}"
                        )
                        
                        if new_status != app['status']:
                            update_application_status(app['id'], new_status)
                            st.rerun()
                        
                        # Delete button
                        if st.button("🗑️ Delete", key=f"delete_{app['id']}", type="secondary"):
                            delete_application(app['id'])
                            st.rerun()
                    
                    # Show cover letter if generated
                    if f'cover_letter_{app["id"]}' in st.session_state:
                        st.text_area(
                            f"Generated Cover Letter for {app['job_title']} at {app['company_name']}",
                            st.session_state[f'cover_letter_{app["id"]}'],
                            height=400,
                            key=f"cover_display_{app['id']}"
                        )
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Cover Letter",
                            data=st.session_state[f'cover_letter_{app["id"]}'],
                            file_name=f"cover_letter_{app['company_name']}_{app['job_title']}.txt",
                            mime="text/plain",
                            key=f"download_{app['id']}"
                        )
                    
                    st.divider()
        else:
            st.info("No applications yet. Use the form above to add your first application!")
    
    elif page == "Analytics":
        st.header("📊 Analytics Dashboard")
        analytics_dashboard()
    
    elif page == "Resume":
        st.header("📄 Resume Information")
        st.write("Add your resume content here. This will be used to generate tailored cover letters.")
        
        resume_content = st.text_area(
            "Resume Content",
            value=st.session_state.resume,
            placeholder="Paste your resume content here, including your experience, skills, education, etc...",
            height=400
        )
        
        if st.button("💾 Save Resume"):
            st.session_state.resume = resume_content
            if save_resume_data():
                st.success("Resume saved successfully!")
            else:
                st.error("Error saving resume. Please try again.")
        
        # Display current resume info
        if st.session_state.resume:
            st.subheader("Current Resume Preview")
            preview_text = st.session_state.resume[:500] + "..." if len(st.session_state.resume) > 500 else st.session_state.resume
            st.text_area("Preview", preview_text, height=200, disabled=True)

    # Sidebar statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Quick Stats")
    if st.session_state.applications:
        df = pd.DataFrame(st.session_state.applications)
        st.sidebar.metric("Total Applications", len(df))
        
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            st.sidebar.write(f"**{status}**: {count}")
    else:
        st.sidebar.write("No applications yet")
    
    # Export data
    st.sidebar.markdown("---")
    if st.session_state.applications:
        df = pd.DataFrame(st.session_state.applications)
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="⬇️ Export to CSV",
            data=csv,
            file_name=f"job_applications_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Data persistence status
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Data Status")
    session_id = get_user_session_id()
    st.sidebar.write(f"Session ID: `{session_id}`")
    st.sidebar.write(f"Apps: {len(st.session_state.applications)}")
    st.sidebar.write(f"Resume: {'✅' if st.session_state.resume else '❌'}")

if __name__ == "__main__":
    main()
