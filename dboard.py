import streamlit as st
import requests
import pandas as pd
from hashlib import sha256
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="FREED", page_icon="https://freed.care/favicon.ico")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style = 'text-align: center; color: blue;'> FREED MARKETING DASHBOARD </h1>", unsafe_allow_html = True)

#For AUTH 
dc_number = st.secrets["moengage"]["dc_number"]
workspace_id = st.secrets["moengage"]["workspace_id"]
secret_key = st.secrets["moengage"]["secret_key"]
password = st.secrets["moengage"]["password"]

# Full Filename
input_filename = st.text_input("Report Filename")

st.markdown("---")
dashboard_type = st.selectbox("Select Dashboard Type", ["Campaign", "Flows"])

if dashboard_type == "Flows":
    flows_channel = st.selectbox("Select Channel (Flows)", ["PUSH", "EMAIL", "WHATSAPP", "CONNECTOR"])
else:
    campaign_channel = st.selectbox("Select Channel (Campaign)", ["PUSH", "EMAIL", "WHATSAPP"])

st.markdown("---")

# -------------------------------------------------------------------------------------------------------
# -----------------------------------Filename (without.zip)----------------------------------------------
# -------------------------------------------------------------------------------------------------------
def normalize_filename(fn):
    if not fn:
        return None
    fn = fn.strip()
    if fn.lower().endswith(".zip"):
        return fn
    # if user passed ReportName_YYYYMMDD, append .zip
    return f"{fn}.zip"

# -------------------------------------------------------------------------------------------------------
# -------------------------------Fetch Response from MoEngage--------------------------------------------
# -------------------------------------------------------------------------------------------------------
def fetch_zip_and_list_files(dc_number, workspace_id, secret_key, password, filename):
    
    try:
        signature_raw = f"{workspace_id}|{filename}|{secret_key}"
        signature = sha256(signature_raw.encode()).hexdigest()

        url = f"https://api-{dc_number}.moengage.com/campaign_reports/rest_api/{workspace_id}/{filename}"
        headers = {
            "Authorization": requests.auth._basic_auth_str(workspace_id, password),
            "Signature": signature
        }

        st.write("📡 Fetching report from MoEngage...")
        res = requests.get(url, headers=headers, timeout=60)

        if res.status_code != 200:
            st.error(f"Error fetching file. HTTP {res.status_code}: {res.text}")
            return None, None

        zip_bytes = BytesIO(res.content)
        zip_obj = zipfile.ZipFile(zip_bytes)
        all_files = zip_obj.namelist()
        return zip_obj, all_files

    except Exception as e:
        st.error(f"Exception while fetching ZIP: {e}")
        return None, None

# -------------------------------------------------------------------------------------------------------
# ------------------------------------LOAD CSV-------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def load_csv_from_zip(zip_ref, file_name):
    try:
        with zip_ref.open(file_name) as f:
            return pd.read_csv(f)
    except Exception as e:
        st.error(f"Failed to load CSV '{file_name}': {e}")
        return None

# ------------------------------------------------------------------------------------------------------
# ------------------------------------Fetch button------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
if st.button("Fetch & Build Dashboard"):
    if not (dc_number and workspace_id and secret_key and password and input_filename):
        st.error("❌ Please Enter Report File Name")
        st.stop()

    filename = normalize_filename(input_filename)
    if not filename:
        st.error("❌ Invalid filename")
        st.stop()

    zip_obj, all_files = fetch_zip_and_list_files(dc_number, workspace_id, secret_key, password, filename)
    if zip_obj is None:
        st.stop()

    st.success("Loaded!")
    
    target_csv = None

    if dashboard_type == "Flows":
        matching_files = [
            f for f in all_files
            if f.startswith("flows/") and flows_channel.upper() in f.upper()
        ]
        if not matching_files:
            st.error(f"No flow CSV found for channel {flows_channel}. Available files: {all_files}")
            st.stop()
        target_csv = matching_files[0]
        if dashboard_type == "Flows":
            st.success(f"Here is your Report for {flows_channel} ")
        elif dashboard_type == "Campaign":
            st.success(f"Here is your Dashboard for {campaign_channel}")

    else:  
        top_level_files = [f for f in all_files if "/" not in f]
        matching_files = [f for f in top_level_files if campaign_channel.upper() in f.upper()]
        if not matching_files:
            st.error(f"No campaign CSV found for channel {campaign_channel}. Available files: {all_files}")
            st.stop()
        target_csv = matching_files[0]
        st.success(f"Here is your Report for {campaign_channel}")

    df = load_csv_from_zip(zip_obj, target_csv)
    if df is None:
        st.stop()

    st.session_state.df = df
    st.session_state.channel = flows_channel if dashboard_type == "Flows" else campaign_channel
    st.session_state.dashboard_type = dashboard_type
    st.session_state.dashboard_ready = True

# ---------------------------
# Dashboard UI 
# ---------------------------
if st.session_state.get("dashboard_ready", False):
    df = st.session_state.df.copy()
    channel = st.session_state.channel
    dashboard_type = st.session_state.dashboard_type

    
    def safe_pct(numerator, denominator):
        try:
            num = float(numerator)
            den = float(denominator)
            if den == 0 or pd.isna(den):
                return 0.0
            return round((num / den) * 100.0, 2)
        except Exception:
            return 0.0
        
    RATE_FORMULAS = {
    "Delivery Rate": ("Total Delivered", "Total Sent"),
    "Read Rate": ("Total Read", "Total Delivered"),
    "Open rate": ("Unique opens", "Total Delivered"),
    "CTR": ("Unique clicks", "Total Delivered"),
    "All Platform Impression Rate": ("All Platform Impressions", "All Platform Sent"),
    "All Platform CTR": ("All Platform Clicks", "All Platform Impressions"),
    "Unsubscribe rate": ("Unsubscribes", "Total Sent")
    }

    def apply_common_filters(df_in, date_col_name, status_col_name, key_prefix="common"):
        df_local = df_in.copy()
        with st.sidebar:
            st.title(":blue[Dashboard Filters]")

            #Report Filter 
            st.markdown("### Report Type")
            report_type = st.selectbox(
                "Select Report Type",
                ["Aggregate", "Day Wise"],
                key=f"{key_prefix}_report_type_{channel}"
            )

            # Date Filter
            st.markdown("### Date")
            date_filter_type = st.selectbox(
                "Select Date Filter Type",
                ["All", "Yesterday", "Last 7 Days", "Last 14 Days", "Last 21 Days", "Custom Range"],
                key=f"{key_prefix}_date_{channel}"
            )

            if date_col_name not in df_local.columns:
                df_local[date_col_name] = None

            df_local[date_col_name] = pd.to_datetime(df_local[date_col_name], errors="coerce")
            today = pd.Timestamp.today().normalize()

            if date_filter_type == "Yesterday":
                yesterday = today - pd.Timedelta(days=1)
                df_local = df_local[df_local[date_col_name].dt.date == yesterday.date()]

            elif date_filter_type == "Last 7 Days":
                last_7 = today - pd.Timedelta(days=7)
                df_local = df_local[(df_local[date_col_name] >= last_7) & (df_local[date_col_name] <= today)]

            elif date_filter_type == "Last 14 Days":
                last_14 = today - pd.Timedelta(days=14)
                df_local = df_local[(df_local[date_col_name] >= last_14) & (df_local[date_col_name] <= today)]

            elif date_filter_type == "Last 21 Days":
                last_21 = today - pd.Timedelta(days=21)
                df_local = df_local[(df_local[date_col_name] >= last_21) & (df_local[date_col_name] <= today)]

            elif date_filter_type == "Custom Range":
                colA, colB = st.columns(2)
                with colA:
                    start_date = st.date_input(
                        "Start Date",
                        df_local[date_col_name].min().date() if df_local[date_col_name].notna().any() else today.date(),
                        key=f"{key_prefix}_start_{channel}"
                    )
                with colB:
                    end_date = st.date_input(
                        "End Date",
                        df_local[date_col_name].max().date() if df_local[date_col_name].notna().any() else today.date(),
                        key=f"{key_prefix}_end_{channel}"
                    )

                if start_date and end_date:
                    df_local = df_local[(df_local[date_col_name].dt.date >= start_date) & (df_local[date_col_name].dt.date <= end_date)]

            ##STATUS FILTER
            st.markdown("### Status")
            status_list = sorted(df_local[status_col_name].dropna().unique())

            selected_statuses = st.multiselect(
                f"Select {status_col_name}",
                options=status_list,
                default=status_list,
                key=f"{key_prefix}_status_{channel}"
            )

            df_local = df_local[df_local[status_col_name].isin(selected_statuses)]

            
            # Flow Name filter (if present)
            flow_col_name = "Flows Name"
            if flow_col_name in df_local.columns:
                st.markdown("### Flow Name")
                flow_list = sorted(df_local[flow_col_name].dropna().unique())
                selected_flow = st.selectbox(
                    f"Select {flow_col_name}",
                    ["All"] + flow_list,
                    key=f"{key_prefix}_flow_{channel}"
                )
                if selected_flow != "All":
                    df_local = df_local[df_local[flow_col_name] == selected_flow]

            # Campaign Name filter

            campaign_col_name = "Campaign Name"
            if campaign_col_name in df_local.columns:
                st.markdown("### Campaign Name")
                campaign_list = sorted(df_local[campaign_col_name].dropna().unique())
                selected_campaign = st.selectbox(
                    f"Select {campaign_col_name}",
                    ["All"] + campaign_list,
                    key=f"{key_prefix}_campaign_{channel}"
                )
                if selected_campaign != "All":
                    df_local = df_local[df_local[campaign_col_name].isin([selected_campaign])]

            return df_local, report_type

   
    if dashboard_type == "Flows" and channel in ["PUSH", "EMAIL", "WHATSAPP", "CONNECTOR"]:
       
        channel_cols = {
            "PUSH": [
                "Campaign Name", "Flows Name", "Date", "Campaign Status",
                "All Platform Attempted", "All Platform Sent", "All Platform Failed",
                "All Platform Impressions", "All Platform Impression Rate",
                "All Platform Clicks", "All Platform CTR",
                "Conversion Goal 1 Event", "Conversion Goal 2 Event"
            ],
            "EMAIL": [
                "Campaign Name", "Flows Name", "Date", "Campaign Status",
                "Total Sent", "Total Delivered", "Total Open", "Unique opens",
                "Open rate", "Total clicks", "Unique clicks", "CTR",
                "Unsubscribes", "Unsubscribe rate"
            ],
            "WHATSAPP": [
                "Campaign Name", "Flows Name", "Date", "Campaign Status",
                "Total Sent", "Total Delivered", "Total Read", "Read Rate", "Delivery Rate",
                "Total clicks", "Unique clicks", "CTR",
                "Conversion Goal 1 Event", "Conversion Goal 2 Event"
            ],
            "CONNECTOR": [
                "Campaign Name", "Flows Name", "Date","Campaign Status", "Attempted", "Sent"
            ]
        }

        required_cols = channel_cols[channel]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        df_filtered, report_type = apply_common_filters(df, "Date", "Campaign Status", key_prefix=f"campaign_{channel}")

        if channel == "PUSH":
            sum_cols = ["All Platform Attempted", "All Platform Sent", "All Platform Failed", "All Platform Impressions", "All Platform Clicks"]
        elif channel == "EMAIL":
            sum_cols = ["Total Sent", "Total Delivered", "Total Open", "Unique opens", "Total clicks", "Unique clicks", "Unsubscribes"]
        elif channel == "WHATSAPP":
            sum_cols = ["Total Sent", "Total Delivered", "Total Read", "Total clicks", "Unique clicks"]
        else:
            sum_cols = ["Attempted", "Sent"]

        for c in sum_cols:
            if c not in df_filtered.columns:
                df_filtered[c] = None
            df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce")

        agg_dict = {c: "sum" for c in sum_cols}
        for c in required_cols:
            if c not in agg_dict and c != "Campaign Name":
                agg_dict[c] = lambda x: x.dropna().astype(str).iloc[0] if x.dropna().shape[0] > 0 else None

        if report_type == "Day Wise":
            if dashboard_type in ["Flows", "Campaigns"]:
                group_keys = ["Campaign Name", "Campaign Status", "Flows Name", "Date"]
        else: 
            group_keys = ["Campaign Name", "Campaign Status", "Flows Name"]

        grouped = df_filtered.groupby(group_keys, as_index=False).agg(agg_dict)
        grouped = grouped.fillna({col: 0 for col in sum_cols})

        if channel == "PUSH":
            grouped["All Platform Impression Rate"] = grouped.apply(lambda r: safe_pct(r.get("All Platform Impressions", 0), r.get("All Platform Sent", 0)), axis=1)
            grouped["All Platform CTR"] = grouped.apply(lambda r: safe_pct(r.get("All Platform Clicks", 0), r.get("All Platform Impressions", 0)), axis=1)
            display_cols = [c for c in required_cols if c in grouped.columns]
            for c in required_cols:
                if c not in grouped.columns:
                    grouped[c] = None
                    display_cols.append(c)
            grouped = grouped[display_cols]
            st.subheader("PUSH Campaigns (Flows)")
            st.dataframe(grouped, use_container_width=True)

        elif channel == "EMAIL":
            grouped["Open rate"] = grouped.apply(lambda r: safe_pct(r.get("Unique opens", 0), r.get("Total Delivered", 0)), axis=1)
            grouped["CTR"] = grouped.apply(lambda r: safe_pct(r.get("Unique clicks", 0), r.get("Total Delivered", 0)), axis=1)
            grouped["Unsubscribe rate"] = grouped.apply(lambda r: safe_pct(r.get("Unsubscribes", 0), r.get("Total Sent", 0)), axis=1)
            display_cols = [c for c in required_cols if c in grouped.columns]
            for c in required_cols:
                if c not in grouped.columns:
                    grouped[c] = None
                    display_cols.append(c)
            grouped = grouped[display_cols]
            st.subheader("EMAIL Campaigns (Flows)")
            st.dataframe(grouped, use_container_width=True)

        elif channel == "WHATSAPP":
            grouped["Delivery Rate"] = grouped.apply(lambda r: safe_pct(r.get("Total Delivered", 0), r.get("Total Sent", 0)), axis=1)
            grouped["Read Rate"] = grouped.apply(lambda r: safe_pct(r.get("Total Read", 0), r.get("Total Delivered", 0)), axis=1)
            grouped["CTR"] = grouped.apply(lambda r: safe_pct(r.get("Unique clicks", 0), r.get("Total Delivered", 0)), axis=1)
            display_cols = [c for c in required_cols if c in grouped.columns]
            for c in required_cols:
                if c not in grouped.columns:
                    grouped[c] = None
                    display_cols.append(c)
            grouped = grouped[display_cols]
            st.subheader("WHATSAPP Campaigns (Flows)")
            st.dataframe(grouped, use_container_width=True)

        else:  # CONNECTOR
            display_cols = [c for c in required_cols if c in grouped.columns]
            for c in required_cols:
                if c not in grouped.columns:
                    grouped[c] = None
                    display_cols.append(c)
            grouped = grouped[display_cols]
            st.subheader("CONNECTOR Campaigns (Flows)")
            st.dataframe(grouped, use_container_width=True)

        
        st.markdown("### Summary")
        if st.button(f"Generate Summary for {channel}"):
            
            summary_cols = [
                col for col in grouped.columns 
                if pd.api.types.is_numeric_dtype(grouped[col]) and "Conversion" not in col
            ]

            summary_data = {}

            totals = {col: grouped[col].sum() for col in summary_cols}

            for k, v in totals.items():
                summary_data[k] = v

            for rate_name, (num_col, den_col) in RATE_FORMULAS.items():
                if num_col in totals and den_col in totals:
                    rate_value = safe_pct(totals[num_col], totals[den_col])
                    summary_data[rate_name] = f"{rate_value:.2f}%"

            st.success(f"Summary for {channel} (Flows)")
            st.dataframe(pd.DataFrame(summary_data.items(), columns=["Metric", "Total"]), use_container_width=True)

    # CAMPAIGN branch
    elif dashboard_type == "Campaign" and channel in ["PUSH", "EMAIL", "WHATSAPP"]:
        
        channel_cols = {
            "PUSH": [
                "Campaign Name", "Date", "Campaign Status",
                "All Platform Attempted", "All Platform Sent", "All Platform Failed",
                "All Platform Impressions", "All Platform Impression Rate",
                "All Platform Clicks", "All Platform CTR",
                "Conversion Goal 1 Event", "Conversion Goal 2 Event"
            ],
            "EMAIL": [
                "Campaign Name", "Date", "Campaign Status",
                "Total Sent", "Total Delivered", "Total Open", "Unique opens",
                "Open rate", "Total clicks", "Unique clicks", "CTR",
                "Unsubscribes", "Unsubscribe rate"
            ],
            "WHATSAPP": [
                "Campaign Name", "Date", "Campaign Status",
                "Total Sent", "Total Delivered", "Total Read", "Read Rate", "Delivery Rate",
                "Total clicks", "Unique clicks", "CTR",
                "Conversion Goal 1 Event", "Conversion Goal 2 Event"
            ],
        }

        required_cols = channel_cols[channel]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        status_column = "Campaign Status"
        df_filtered, report_type = apply_common_filters(df, "Date", status_column, key_prefix=f"campaign_{channel}")

        if channel == "PUSH":
            sum_cols = ["All Platform Attempted", "All Platform Sent", "All Platform Failed", "All Platform Impressions", "All Platform Clicks"]
        elif channel == "EMAIL":
            sum_cols = ["Total Sent", "Total Delivered", "Total Open", "Unique opens", "Total clicks", "Unique clicks", "Unsubscribes"]
        elif channel == "WHATSAPP":
            sum_cols = ["Total Sent", "Total Delivered", "Total Read", "Total clicks", "Unique clicks"]
        else:
            sum_cols = ["Attempted", "Sent"]

        for c in sum_cols:
            if c not in df_filtered.columns:
                df_filtered[c] = None
            df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce")

        agg_dict = {c: "sum" for c in sum_cols}
        for c in required_cols:
            if c not in agg_dict and c != "Campaign Name":
                agg_dict[c] = lambda x: x.dropna().astype(str).iloc[0] if x.dropna().shape[0] > 0 else None

        if report_type == "Day Wise":
            group_keys = ["Campaign Name", "Campaign Status", "Date"]
        else:
            group_keys = ["Campaign Name", "Campaign Status"]

        grouped = df_filtered.groupby(group_keys, as_index=False).agg(agg_dict)
        grouped = grouped.fillna({col: 0 for col in sum_cols})

        if channel == "PUSH":
            grouped["All Platform Impression Rate"] = grouped.apply(lambda r: safe_pct(r.get("All Platform Impressions", 0), r.get("All Platform Sent", 0)), axis=1)
            grouped["All Platform CTR"] = grouped.apply(lambda r: safe_pct(r.get("All Platform Clicks", 0), r.get("All Platform Impressions", 0)), axis=1)
            display_cols = [c for c in required_cols if c in grouped.columns]
            for c in required_cols:
                if c not in grouped.columns:
                    grouped[c] = None
                    display_cols.append(c)
            grouped = grouped[display_cols]
            st.subheader("PUSH Campaigns")
            st.dataframe(grouped, use_container_width=True)

        elif channel == "EMAIL":
            grouped["Open rate"] = grouped.apply(lambda r: safe_pct(r.get("Unique opens", 0), r.get("Total Delivered", 0)), axis=1)
            grouped["CTR"] = grouped.apply(lambda r: safe_pct(r.get("Unique clicks", 0), r.get("Total Delivered", 0)), axis=1)
            grouped["Unsubscribe rate"] = grouped.apply(lambda r: safe_pct(r.get("Unsubscribes", 0), r.get("Total Sent", 0)), axis=1)
            display_cols = [c for c in required_cols if c in grouped.columns]
            for c in required_cols:
                if c not in grouped.columns:
                    grouped[c] = None
                    display_cols.append(c)
            grouped = grouped[display_cols]
            st.subheader("EMAIL Campaigns")
            st.dataframe(grouped, use_container_width=True)

        elif channel == "WHATSAPP":
            grouped["Delivery Rate"] = grouped.apply(lambda r: safe_pct(r.get("Total Delivered", 0), r.get("Total Sent", 0)), axis=1)
            grouped["Read Rate"] = grouped.apply(lambda r: safe_pct(r.get("Total Read", 0), r.get("Total Delivered", 0)), axis=1)
            grouped["CTR"] = grouped.apply(lambda r: safe_pct(r.get("Unique clicks", 0), r.get("Total Delivered", 0)), axis=1)
            display_cols = [c for c in required_cols if c in grouped.columns]
            for c in required_cols:
                if c not in grouped.columns:
                    grouped[c] = None
                    display_cols.append(c)
            grouped = grouped[display_cols]
            st.subheader("WHATSAPP Campaigns")
            st.dataframe(grouped, use_container_width=True)

        st.markdown("### Summary")
        if st.button(f"Generate Summary for {channel}"):
            
            summary_cols = [
                col for col in grouped.columns 
                if pd.api.types.is_numeric_dtype(grouped[col]) and "Conversion" not in col
            ]

            summary_data = {}

            totals = {col: grouped[col].sum() for col in summary_cols}

            for k, v in totals.items():
                summary_data[k] = v

            for rate_name, (num_col, den_col) in RATE_FORMULAS.items():
                if num_col in totals and den_col in totals:
                    rate_value = safe_pct(totals[num_col], totals[den_col])
                    summary_data[rate_name] = f"{rate_value:.2f}%"

            st.success(f"Summary for {channel} Campaigns")
            st.dataframe(pd.DataFrame(summary_data.items(), columns=["Metric", "Total"]), use_container_width=True)