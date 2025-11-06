import pandas as pd
import numpy as np
import io
import streamlit as st
from datetime import datetime

# --- Core Utility Functions (Adapted for Streamlit) ---

@st.cache_data
def load_and_clean_data(uploaded_file, file_name):
    """
    Loads data from an uploaded file, removes entirely empty columns,
    and converts column names to a consistent, stripped format.
    """
    st.write(f"Loading data from: {file_name}")
    try:
        # Determine file type based on the name from Streamlit
        if file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.csv'):
            try:
                # Try common encodings
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        else:
            st.error(f"Unsupported file type: {file_name}")
            return pd.DataFrame()

        # 1. Strip whitespace from column names for robust handling
        df.columns = df.columns.str.strip()

        # 2. Identify and remove columns that are entirely NaN (empty)
        df = df.dropna(axis=1, how='all')

        st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_name}.")
        return df
    except Exception as e:
        st.error(f"An error occurred while loading {file_name}: {e}")
        return pd.DataFrame()

def parse_date_column(df, date_col_name):
    # ... (content of parse_date_column)
    if date_col_name not in df.columns:
        return df
        
    df[date_col_name] = df[date_col_name].astype(str)

    date_formats = [
        '%Y-%m-%d %H:%M:%S', 
        '%Y-%m-%d %H:%M',
        '%m/%d/%Y %H:%M',
        '%d-%b-%y %H:%M:%S', 
        '%Y-%m-%dT%H:%M:%S.%fZ', 
        '%Y-%m-%d'
    ]
    
    for fmt in date_formats:
        try:
            temp_series = pd.to_datetime(df[date_col_name], format=fmt, errors='coerce')
            if temp_series.notna().sum() > len(df) * 0.5:
                df[date_col_name] = temp_series
                return df
        except ValueError:
            continue
            
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
    return df

def get_registrations(df_userlist):
    # ... (content of get_registrations)
    if df_userlist.empty or 'Joined On' not in df_userlist.columns or 'Email Id' not in df_userlist.columns:
        st.warning("Registration sheet missing required columns: 'Joined On' or 'Email Id'.")
        return pd.DataFrame({'Date': [], 'Type': [], 'Email': []})

    df = df_userlist[['Joined On', 'Email Id']].copy()
    df.columns = ['Date', 'Email']
    df = parse_date_column(df, 'Date')
    df = df.dropna(subset=['Date', 'Email'])
    
    df = df.sort_values('Date').drop_duplicates(subset='Email', keep='first')
    df['Type'] = 'Registered'
    return df[['Date', 'Type', 'Email']] 

def get_subscriptions(df_userlist, df_stripe, df_paystack):
    # ... (content of get_subscriptions)
    all_subscriptions = []

    # 1. Userlist Data
    if not df_userlist.empty and 'User Type' in df_userlist.columns and 'Joined On' in df_userlist.columns and 'Email Id' in df_userlist.columns:
        df_userlist_sub = df_userlist[
            df_userlist['User Type'].astype(str).str.contains('sub', case=False, na=False)
        ].copy()
        
        if not df_userlist_sub.empty:
            df_userlist_sub = df_userlist_sub[['Joined On', 'Email Id']].copy()
            df_userlist_sub.columns = ['Date', 'Email']
            df_userlist_sub = parse_date_column(df_userlist_sub, 'Date')
            df_userlist_sub['Source'] = 'Userlist'
            all_subscriptions.append(df_userlist_sub)
            st.code(f"DEBUG: Found {len(df_userlist_sub)} potential subscribed users from Userlist.")
        else:
            st.code("DEBUG: Found 0 subscribed users from Userlist.")
    
    # 2. Stripe Data (unified_customers.csv)
    if not df_stripe.empty and 'Status' in df_stripe.columns and 'Created (UTC)' in df_stripe.columns and 'Email' in df_stripe.columns:
        df_stripe_sub = df_stripe[
            (df_stripe['Status'].astype(str).str.lower().isin(['active', 'trialing'])) | 
            (~df_stripe['Status'].astype(str).str.lower().isin(['cancelled', 'failed'])) |
            df_stripe['Status'].isna()
        ].copy()
        
        if not df_stripe_sub.empty:
            df_stripe_sub = df_stripe_sub[['Created (UTC)', 'Email']].copy()
            df_stripe_sub.columns = ['Date', 'Email']
            df_stripe_sub = parse_date_column(df_stripe_sub, 'Date')
            df_stripe_sub['Source'] = 'Stripe'
            all_subscriptions.append(df_stripe_sub)
            st.code(f"DEBUG: Found {len(df_stripe_sub)} potential subscribed users from Stripe.")
        else:
            st.code("DEBUG: Found 0 subscribed users from Stripe.")
    
    # 3. Paystack Data
    if not df_paystack.empty and 'Gateway Response' in df_paystack.columns and 'Date' in df_paystack.columns and 'Customer (email)' in df_paystack.columns:
        df_paystack_sub = df_paystack[
            (df_paystack['Gateway Response'] == 'Successful') & 
            df_paystack['Customer (email)'].notna()
        ].copy()
        
        if not df_paystack_sub.empty:
            df_paystack_sub = df_paystack_sub[['Date', 'Customer (email)']].copy()
            df_paystack_sub.columns = ['Date', 'Email']
            df_paystack_sub = parse_date_column(df_paystack_sub, 'Date')
            df_paystack_sub['Source'] = 'Paystack'
            all_subscriptions.append(df_paystack_sub)
            st.code(f"DEBUG: Found {len(df_paystack_sub)} potential subscribed users from Paystack.")
        else:
            st.code("DEBUG: Found 0 subscribed users from Paystack.")

    if not all_subscriptions:
        return pd.DataFrame({'Date': [], 'Type': []})

    # 4. Combine and Clean
    df_combined = pd.concat(all_subscriptions, ignore_index=True)
    
    rows_before_drop = len(df_combined)
    df_combined = df_combined.dropna(subset=['Date', 'Email'])
    rows_after_drop = len(df_combined)

    if rows_before_drop != rows_after_drop:
        st.warning(f"DEBUG: Dropped {rows_before_drop - rows_after_drop} rows due to invalid/missing date or email during consolidation.")

    df_combined = df_combined.sort_values('Date').drop_duplicates(subset='Email', keep='first')
    
    df_combined['Type'] = 'Subscribed'
    return df_combined[['Date', 'Type']]

# --- NEW SUMMARY FUNCTION ---

def generate_summary(excel_output):
    """Reads the generated Excel sheets and extracts the required metrics."""
    try:
        # Read the Excel output buffer
        day_df = pd.read_excel(excel_output, sheet_name='Day', index_col=0)
        month_df = pd.read_excel(excel_output, sheet_name='Month', index_col=0)
        year_df = pd.read_excel(excel_output, sheet_name='Year', index_col=0)
        
        # Helper to safely get the last row's data (latest period)
        def get_latest_metrics(df):
            if df.empty:
                return 0, 0, "N/A"
            latest_row = df.iloc[-1]
            date_col = df.index.name
            date_str = str(latest_row.name).split(' ')[0] # Get date part only
            return int(latest_row.get('Registered', 0)), int(latest_row.get('Subscribed', 0)), date_str

        # Metrics for Today
        reg_today, sub_today, date_today = get_latest_metrics(day_df)
        
        # Metrics for This Month
        reg_month, sub_month, _ = get_latest_metrics(month_df)
        
        # Metrics for This Year
        reg_year, sub_year, _ = get_latest_metrics(year_df)
        
        # Format the summary string
        summary = f"""
Date: {date_today}
Time: {datetime.now().strftime('%H:%M:%S')}
Number of Registrations Today: {reg_today}
Number of Active Subscribers Today: {sub_today}
Number of Registrations this month: {reg_month}
Number of Active Subscribers this month: {sub_month}
Number of Registrations this year: {reg_year}
Number of Active Subscribers year: {sub_year}
"""
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Could not generate summary."


def aggregate_and_merge(df_registrations, df_subscriptions):
    """
    Aggregates registration and subscription data by time period and merges them.
    Saves the results to a multi-sheet Excel file buffer.
    """
    df_reg_final = df_registrations[['Date', 'Type']]
    df_sub_final = df_subscriptions[['Date', 'Type']]
    
    df_events = pd.concat([df_reg_final, df_sub_final], ignore_index=True)
    df_events = df_events.dropna(subset=['Date']) 

    time_periods = {
        'Hour': 'H',
        'Day': 'D',
        'Week': 'W',
        'Month': 'M',
        'Year': 'Y'
    }

    # Use a BytesIO buffer instead of a file path for Streamlit download
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, freq in time_periods.items():
            
            # Using the recommended frequency aliases to silence FutureWarnings
            if freq == 'H': freq = 'h'
            elif freq == 'M': freq = 'ME'
            elif freq == 'Y': freq = 'YE'
            
            df_agg = df_events.groupby(
                [pd.Grouper(key='Date', freq=freq), 'Type']
            )['Type'].count().unstack(fill_value=0)
            
            if 'Registered' not in df_agg.columns:
                df_agg['Registered'] = 0
            if 'Subscribed' not in df_agg.columns:
                df_agg['Subscribed'] = 0
            
            df_agg = df_agg[['Registered', 'Subscribed']]
            df_agg.index.name = f'{name}_Starting_Period'
            df_agg.columns.name = None
            df_agg.index = df_agg.index.astype(str)
            df_agg.to_excel(writer, sheet_name=name)
            
    output.seek(0)
    return output


# --- Streamlit UI and Execution ---

st.set_page_config(page_title="User Analytics Report Generator", layout="centered")
st.title("📊 User Analytics Report Generator")
st.markdown("Upload the three required files and click **'Generate Report'** to get the multi-sheet Excel output.")

# File upload widgets
st.subheader("1. Upload Data Files")
userlist_file = st.file_uploader(
    "Userlist (userlist_20251106.xlsx)", 
    type=['xlsx', 'csv'], 
    key='userlist'
)
stripe_file = st.file_uploader(
    "Stripe/Unified Customers (unified_customers.csv)", 
    type=['csv'], 
    key='stripe'
)
paystack_file = st.file_uploader(
    "Paystack Transactions (EbonyLife_ON_transactions_...csv)", 
    type=['csv'], 
    key='paystack'
)

# Run button
if st.button("▶️ Generate Report", type="primary"):
    
    if not all([userlist_file, stripe_file, paystack_file]):
        st.error("Please upload all three required files to proceed.")
    else:
        st.header("--- Analysis Log ---")
        
        # Load DataFrames
        df_userlist = load_and_clean_data(userlist_file, userlist_file.name)
        df_stripe = load_and_clean_data(stripe_file, stripe_file.name)
        df_paystack = load_and_clean_data(paystack_file, paystack_file.name)
        
        if df_userlist.empty or df_stripe.empty or df_paystack.empty:
            st.error("Cannot proceed. One or more files failed to load correctly.")
        else:
            with st.spinner("Processing data and generating report..."):
                
                # 1. Extract Registrations
                df_registrations = get_registrations(df_userlist)
                st.info(f"✅ Found {len(df_registrations)} unique registered users.")
                
                # 2. Extract Subscriptions
                df_subscriptions = get_subscriptions(df_userlist, df_stripe, df_paystack) 
                st.success(f"Final Count: Found {len(df_subscriptions)} unique subscribed users.")
                
                # 3. Aggregate and Merge
                if df_registrations.empty and df_subscriptions.empty:
                    st.error("🚫 No valid data to analyze.")
                else:
                    excel_output = aggregate_and_merge(df_registrations, df_subscriptions)
                    
                    st.header("🎉 Report Complete!")
                    st.balloons()
                    
                    # 4. Generate and display the summary
                    summary_text = generate_summary(excel_output)
                    
                    st.subheader("📈 Daily Metrics Summary")
                    st.code(summary_text)

                    # 5. Provide download button
                    st.download_button(
                        label="⬇️ Download user_analysis_report.xlsx",
                        data=excel_output,
                        file_name='user_analysis_report.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                    # Optional: Display a sample of the daily data
                    st.subheader("Sample Daily Aggregation:")
                    daily_data_reader = pd.read_excel(excel_output, sheet_name='Day')
                    st.dataframe(daily_data_reader.head())
