"""
Talent Match Intelligence System - Step 3: AI-Powered Dashboard
Complete Streamlit Application with Supabase Integration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client, Client
import os
from datetime import datetime
import json
import requests

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Talent Match Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SUPABASE CONNECTION
# ==========================================
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    url = os.getenv("SUPABASE_URL", "https://pqbirtpyunlibganxewl.supabase.co")
    key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBxYmlydHB5dW5saWJnYW54ZXdsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE1Mzk0MTUsImV4cCI6MjA3NzExNTQxNX0.1z1GarIGB7R2tB1NOPCj3upgj-gSctXmZo1XJHRkcAk")
    return create_client(url, key)

supabase: Client = init_supabase()

# ==========================================
# AI GENERATION FUNCTIONS
# ==========================================
def generate_job_profile_with_ai(role_name, job_level, role_purpose, benchmark_employees_data):
    """Generate job requirements using AI (OpenRouter/any LLM)"""
    
    # Simple prompt for AI
    prompt = f"""
You are an HR analyst. Generate a job profile for:

Role: {role_name}
Level: {job_level}
Purpose: {role_purpose}

Based on {len(benchmark_employees_data)} benchmark employees who are top performers.

Generate:
1. Key Requirements (3-5 bullet points)
2. Job Description (2-3 sentences)
3. Core Competencies (5-7 competencies)

Format as JSON:
{{
    "requirements": ["req1", "req2", ...],
    "description": "...",
    "competencies": ["comp1", "comp2", ...]
}}
"""
    
    try:
        # Option 1: Use OpenRouter (free tier)
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Try to parse JSON from response
            try:
                return json.loads(content)
            except:
                # Fallback if not proper JSON
                return {
                    "requirements": ["Advanced analytical skills", "Stakeholder management", "Strategic thinking"],
                    "description": content[:200],
                    "competencies": ["Quality Delivery", "Future Thinking", "Leadership"]
                }
    except Exception as e:
        st.warning(f"AI generation failed: {e}. Using template.")
    
    # Fallback: Template-based generation
    return {
        "requirements": [
            f"Experience in {role_name.lower()} or related field",
            f"Proven track record at {job_level} level",
            "Strong analytical and problem-solving skills",
            "Excellent stakeholder management",
            "Strategic thinking and execution"
        ],
        "description": f"This role focuses on {role_purpose}. The ideal candidate will have demonstrated excellence across quality delivery, strategic capability, and cultural alignment.",
        "competencies": [
            "Quality-Driven Delivery (QDD)",
            "Future Thinking & Change (FTC)",
            "Strategic Operations (STO)",
            "Self-Awareness & EQ (SEA)",
            "Values & Culture (VCU)"
        ]
    }

# ==========================================
# DATA FETCHING FUNCTIONS
# ==========================================
@st.cache_data(ttl=300)
def get_rating_5_employees():
    """Get all employees with high ratings for benchmark selection"""
    try:
        # Method 1: Try using the view first
        try:
            response = supabase.table('performance_yearly_with_employees') \
                .select('employee_id, fullname, nip, rating') \
                .gte('rating', 4) \
                .order('rating', desc=True) \
                .order('year', desc=True) \
                .execute()
            
            if response.data and len(response.data) > 0:
                df = pd.DataFrame(response.data)
                df_unique = df.drop_duplicates(subset=['employee_id'], keep='first')
                return df_unique[['employee_id', 'fullname', 'nip', 'rating']]
        except Exception as view_error:
            st.warning(f"View method failed: {view_error}. Trying alternative method...")
        
        # Method 2: Fetch separately without view
        # Step 1: Get performance data
        perf_response = supabase.table('performance_yearly') \
            .select('employee_id, rating, year') \
            .gte('rating', 4) \
            .order('rating', desc=True) \
            .order('year', desc=True) \
            .execute()
        
        if not perf_response.data:
            st.warning("⚠️ No performance records with rating ≥ 4 found")
            return pd.DataFrame()
        
        # Get unique employee IDs with their best rating
        perf_df = pd.DataFrame(perf_response.data)
        perf_df = perf_df.sort_values(['rating', 'year'], ascending=[False, False])
        perf_df = perf_df.drop_duplicates(subset=['employee_id'], keep='first')
        
        employee_ids = perf_df['employee_id'].tolist()
        
        # Step 2: Get employee details
        emp_response = supabase.table('employees') \
            .select('employee_id, fullname, nip') \
            .in_('employee_id', employee_ids) \
            .execute()
        
        if not emp_response.data:
            st.warning("⚠️ Employee details not found")
            return pd.DataFrame()
        
        # Step 3: Merge data
        emp_df = pd.DataFrame(emp_response.data)
        result_df = emp_df.merge(
            perf_df[['employee_id', 'rating']], 
            on='employee_id', 
            how='left'
        )
        
        st.success(f"✅ Found {len(result_df)} high-performing employees")
        return result_df[['employee_id', 'fullname', 'nip', 'rating']]
        
    except Exception as e:
        st.error(f"Error fetching employees: {str(e)}")
        st.info("💡 Debug info - Check these tables exist:")
        st.code("SELECT * FROM public.performance_yearly LIMIT 5;\nSELECT * FROM public.employees LIMIT 5;")
        return pd.DataFrame()

def run_matching_query(job_vacancy_id):
    """Execute the complete matching SQL query"""
    try:
        # FIXED: Removed schema prefix - now uses public.talent_match_results view
        response = supabase.table('talent_match_results') \
            .select('*') \
            .eq('job_vacancy_id', job_vacancy_id) \
            .execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            return df
        else:
            st.warning("No results found for this vacancy. Make sure to refresh the materialized view after creating new vacancies.")
            st.code("REFRESH MATERIALIZED VIEW case_studyda.talent_match_results;")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching results: {str(e)}")
        
        # Try RPC fallback
        try:
            response = supabase.rpc('match_candidates', {'vacancy_id': job_vacancy_id}).execute()
            if response.data:
                return pd.DataFrame(response.data)
        except:
            pass
        
        st.info("💡 If talent_match_results doesn't exist yet, you need to create the matching query first.")
        return pd.DataFrame()

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def create_radar_chart(employee_data):
    """Create radar chart for TGV comparison"""
    categories = ['Execution Excellence', 'Strategic Capability', 'People & Culture']
    values = [
        employee_data.get('execution_match', 0),
        employee_data.get('strategic_match', 0),
        employee_data.get('culture_match', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the radar
        theta=categories + [categories[0]],
        fill='toself',
        name=employee_data.get('fullname', 'Candidate'),
        line_color='rgb(59, 130, 246)'
    ))
    
    # Add benchmark line at 100%
    fig.add_trace(go.Scatterpolar(
        r=[100, 100, 100, 100],
        theta=categories + [categories[0]],
        fill='toself',
        name='Benchmark (100%)',
        line_color='rgba(16, 185, 129, 0.3)',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 120]
            )
        ),
        showlegend=True,
        height=400
    )
    
    return fig

def create_tv_heatmap(df_results):
    """Create heatmap of TV match rates across candidates"""
    
    # Get top 10 candidates
    top_candidates = df_results.nsmallest(10, 'final_match_rate', keep='first')
    
    # Pivot to get TV match rates
    pivot_data = top_candidates.pivot_table(
        index='fullname',
        columns='tv_name',
        values='tv_match_rate',
        aggfunc='first'
    )
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Competency (TV)", y="Candidate", color="Match %"),
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=50,
        zmax=110
    )
    
    fig.update_layout(height=500)
    return fig

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.title("🎯 Talent Match Intelligence System")
    st.markdown("**AI-Powered Talent Matching for Strategic Hiring**")
    
    # Sidebar Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📝 Create New Vacancy", "📊 View Matching Results", "📈 Analytics Dashboard"]
    )
    
    # Add cache clear button in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Clear Cache & Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # ==========================================
    # PAGE 1: CREATE NEW VACANCY
    # ==========================================
    if page == "📝 Create New Vacancy":
        st.header("Create New Job Vacancy")
        
        with st.form("vacancy_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                role_name = st.text_input("Role Name*", placeholder="e.g., Senior Business Analyst")
                job_level = st.selectbox("Job Level*", ["Grade III", "Grade IV", "Grade V", "Grade VI"])
            
            with col2:
                job_vacancy_id = st.text_input(
                    "Vacancy ID*", 
                    value=f"VAC-{datetime.now().strftime('%Y%m%d-%H%M')}",
                    help="Auto-generated or custom"
                )
                created_by = st.text_input("Created By", placeholder="Your name")
            
            role_purpose = st.text_area(
                "Role Purpose*",
                placeholder="1-2 sentence summary of the role's main objective",
                height=100
            )
            
            st.subheader("Select Benchmark Employees")
            st.markdown("Choose 3-10 employees with **Rating 5** who represent the ideal profile for this role.")
            
            # Get Rating 5 employees
            rating_5_df = get_rating_5_employees()
            
            if not rating_5_df.empty:
                # Multi-select for benchmark employees
                selected_benchmarks = st.multiselect(
                    "Select Benchmark Employees",
                    options=rating_5_df['employee_id'].tolist(),
                    format_func=lambda x: f"{rating_5_df[rating_5_df['employee_id']==x]['fullname'].values[0]} ({x}) - Rating: {rating_5_df[rating_5_df['employee_id']==x]['rating'].values[0]}",
                    help="Select 3-10 top performers (Rating 4-5)"
                )
                
                if selected_benchmarks:
                    st.success(f"✅ {len(selected_benchmarks)} benchmark employees selected")
            else:
                st.error("No high-performing employees (rating ≥4) found in database!")
                st.info("💡 Check your database by running the diagnostic SQL query")
                selected_benchmarks = []
            
            submitted = st.form_submit_button("Create Vacancy & Generate Profile", type="primary")
            
            if submitted:
                if not all([role_name, job_level, role_purpose, selected_benchmarks]):
                    st.error("Please fill all required fields and select benchmark employees!")
                elif len(selected_benchmarks) < 3:
                    st.error("Please select at least 3 benchmark employees!")
                else:
                    with st.spinner("Creating vacancy and generating AI profile..."):
                        try:
                            # Prepare data for insertion
                            insert_data = {
                                'job_vacancy_id': job_vacancy_id,
                                'role_name': role_name,
                                'job_level': job_level,
                                'role_purpose': role_purpose,
                                'selected_talent_ids': selected_benchmarks
                            }
                            
                            # Only add created_by if user provided it
                            if created_by:
                                insert_data['created_by'] = created_by
                            
                            # Insert into talent_benchmarks table
                            response = supabase.table('talent_benchmarks').insert(insert_data).execute()
                            
                            # Generate AI profile
                            benchmark_data = rating_5_df[rating_5_df['employee_id'].isin(selected_benchmarks)]
                            ai_profile = generate_job_profile_with_ai(
                                role_name, job_level, role_purpose, benchmark_data.to_dict('records')
                            )
                            
                            st.success(f"✅ Vacancy created: {job_vacancy_id}")
                            
                            # Display AI-generated profile
                            st.subheader("🤖 AI-Generated Job Profile")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Job Requirements:**")
                                for req in ai_profile['requirements']:
                                    st.markdown(f"- {req}")
                            
                            with col2:
                                st.markdown("**Core Competencies:**")
                                for comp in ai_profile['competencies']:
                                    st.markdown(f"- {comp}")
                            
                            st.markdown("**Job Description:**")
                            st.info(ai_profile['description'])
                            
                            st.markdown("---")
                            st.info("💡 **Next Step:** Go to 'View Matching Results' to see ranked candidates!")
                            
                        except Exception as e:
                            st.error(f"Error creating vacancy: {str(e)}")
    
    # ==========================================
    # PAGE 2: VIEW MATCHING RESULTS
    # ==========================================
    elif page == "📊 View Matching Results":
        st.header("Talent Matching Results")
        
        # Get all vacancies
        vacancies_response = supabase.table('talent_benchmarks').select('*').execute()
        
        if vacancies_response.data:
            vacancies_df = pd.DataFrame(vacancies_response.data)
            
            selected_vacancy = st.selectbox(
                "Select Vacancy",
                vacancies_df['job_vacancy_id'].tolist(),
                format_func=lambda x: f"{vacancies_df[vacancies_df['job_vacancy_id']==x]['role_name'].values[0]} ({x})"
            )
            
            if st.button("🔍 Run Matching Analysis", type="primary"):
                with st.spinner("Running matching algorithm..."):
                    # Fetch results
                    results_df = run_matching_query(selected_vacancy)
                    
                    if not results_df.empty:
                        # Store in session state
                        st.session_state['results_df'] = results_df
                        st.session_state['selected_vacancy_id'] = selected_vacancy
                        st.success(f"✅ Found {results_df['employee_id'].nunique()} candidates")
                    else:
                        st.error("No results found. Please run the SQL query first!")
            
            # Display results if available
            if 'results_df' in st.session_state:
                results_df = st.session_state['results_df']
                
                # Get summary (one row per employee)
                summary_df = results_df.groupby('employee_id').agg({
                    'fullname': 'first',
                    'position': 'first',
                    'grade': 'first',
                    'directorate': 'first',
                    'final_match_rate': 'first',
                    'execution_match': 'first',
                    'strategic_match': 'first',
                    'culture_match': 'first',
                    'match_category': 'first',
                    'recommendation': 'first'
                }).reset_index()
                
                summary_df = summary_df.sort_values('final_match_rate', ascending=False)
                
                # Top candidates overview
                st.subheader("📊 Top 10 Candidates")
                
                top_10 = summary_df.head(10)
                
                for idx, row in top_10.iterrows():
                    with st.expander(f"#{idx+1} - {row['fullname']} | Match: {row['final_match_rate']:.1f}% | {row['match_category']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Overall Match", f"{row['final_match_rate']:.1f}%")
                            st.write(f"**Position:** {row['position']}")
                            st.write(f"**Grade:** {row['grade']}")
                            st.write(f"**Directorate:** {row['directorate']}")
                        
                        with col2:
                            st.metric("Execution", f"{row['execution_match']:.1f}%")
                            st.metric("Strategic", f"{row['strategic_match']:.1f}%")
                            st.metric("Culture", f"{row['culture_match']:.1f}%")
                        
                        with col3:
                            # Get detailed TV data for this employee
                            employee_tv_data = results_df[results_df['employee_id'] == row['employee_id']]
                            
                            # Strengths
                            strengths = employee_tv_data[employee_tv_data['tv_match_rate'] >= 95]['tv_label'].tolist()
                            if strengths:
                                st.markdown("**💪 Strengths:**")
                                for s in strengths[:3]:
                                    st.markdown(f"- {s}")
                            
                            # Development areas
                            gaps = employee_tv_data[employee_tv_data['tv_match_rate'] < 90]['tv_label'].tolist()
                            if gaps:
                                st.markdown("**📈 Development:**")
                                for g in gaps[:3]:
                                    st.markdown(f"- {g}")
                        
                        # Radar chart
                        st.plotly_chart(
                            create_radar_chart(row.to_dict()),
                            use_container_width=True,
                            key=f"radar_{row['employee_id']}"
                        )
                        
                        st.info(f"**Recommendation:** {row['recommendation']}")
                
                # Download results
                st.markdown("---")
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=summary_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"talent_match_{selected_vacancy}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("No vacancies found. Create a new vacancy first!")
    
    # ==========================================
    # PAGE 3: ANALYTICS DASHBOARD
    # ==========================================
    elif page == "📈 Analytics Dashboard":
        st.header("Analytics Dashboard")
        
        if 'results_df' in st.session_state:
            results_df = st.session_state['results_df']
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_candidates = results_df['employee_id'].nunique()
                st.metric("Total Candidates", total_candidates)
            
            with col2:
                excellent_count = results_df[results_df['final_match_rate'] >= 95]['employee_id'].nunique()
                st.metric("Excellent Matches (≥95%)", excellent_count)
            
            with col3:
                good_count = results_df[
                    (results_df['final_match_rate'] >= 85) & 
                    (results_df['final_match_rate'] < 95)
                ]['employee_id'].nunique()
                st.metric("Good Matches (85-94%)", good_count)
            
            with col4:
                avg_match = results_df.groupby('employee_id')['final_match_rate'].first().mean()
                st.metric("Avg Match Rate", f"{avg_match:.1f}%")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Match Rate Distribution")
                summary_df = results_df.groupby('employee_id')['final_match_rate'].first().reset_index()
                fig = px.histogram(
                    summary_df,
                    x='final_match_rate',
                    nbins=20,
                    title="Distribution of Final Match Rates"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("TGV Category Performance")
                tgv_avg = results_df.groupby('tgv_name')['tgv_match_rate'].mean().reset_index()
                fig = px.bar(
                    tgv_avg,
                    x='tgv_name',
                    y='tgv_match_rate',
                    title="Average Match Rate by TGV Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            st.subheader("Competency Heatmap (Top 10 Candidates)")
            st.plotly_chart(create_tv_heatmap(results_df), use_container_width=True)
            
        else:
            st.info("Run matching analysis first to see analytics!")

# ==========================================
# RUN APP
# ==========================================
if __name__ == "__main__":
    main()