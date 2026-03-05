# =============================================================
# dashboard/app.py
# Medicaid Fraud Detection — Interactive Plotly Dash Dashboard
#
# HOW TO RUN:
#   1. First run the Jupyter notebook to populate the database
#   2. Open a Command Prompt in the project root folder
#   3. Run: python dashboard/app.py
#   4. Open browser to: http://127.0.0.1:8050
#
# WHAT THIS FILE DOES:
#   Plotly Dash is a Python framework that builds web apps without
#   writing HTML or JavaScript. Everything is written in Python.
#   The app reads fraud results from MySQL and presents them
#   in an interactive dashboard with toggleable algorithm views.
# =============================================================

import os
import sys
import configparser
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote_plus

# =============================================================
# STEP 1: LOAD CONFIGURATION
# =============================================================

# Build the path to config.ini (one folder up from dashboard/)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', 'config.ini')

if not os.path.exists(CONFIG_FILE):
    print(f'ERROR: config.ini not found at {os.path.abspath(CONFIG_FILE)}')
    print('Please copy config.ini.example to config.ini and fill in your password.')
    sys.exit(1)

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

DB_HOST = config['mysql']['host']
DB_USER = config['mysql']['user']
DB_PASS = config['mysql']['password']
DB_NAME = config['mysql']['database']
DB_PORT = config['mysql'].get('port', '3306')

# Create the SQLAlchemy engine
engine = create_engine(
    f'mysql+mysqlconnector://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}',
    echo=False
)
# =============================================================
# STEP 2: LOAD DATA FROM MYSQL
# =============================================================
# We load the data ONCE when the app starts, not on every user interaction.
# This is called "data caching" — loading upfront makes the dashboard fast.

print('Loading data from MySQL...')

try:
    # Load provider summary with all fraud flags
    provider_df = pd.read_sql('SELECT * FROM provider_summary', engine)

    # Load fraud flags detail
    flags_df = pd.read_sql('SELECT * FROM fraud_flags', engine)

    print(f'✅ Loaded {len(provider_df):,} providers and {len(flags_df):,} fraud flags')

except Exception as e:
    print(f'ERROR loading data from MySQL: {e}')
    print('Make sure you have run the Jupyter notebook first to populate the database.')
    sys.exit(1)

# =============================================================
# STEP 3: CONSTANTS & HELPERS
# =============================================================

# Algorithm metadata — used to build the toggle buttons and filter data
ALGORITHMS = {
    'zscore': {
        'label': 'Z-Score / IQR',
        'flag_col': 'is_outlier_zscore',
        'color': '#3B82F6',   # Blue
        'description': 'Flags providers whose billing amounts are statistically far from the mean (>3 standard deviations).',
        'icon': '📊'
    },
    'isolation_forest': {
        'label': 'Isolation Forest',
        'flag_col': 'is_outlier_isolation_forest',
        'color': '#10B981',   # Green
        'description': 'ML algorithm that detects anomalies by isolating points that are easiest to separate from the rest of the data.',
        'icon': '🌲'
    },
    'dbscan': {
        'label': 'DBSCAN Clustering',
        'flag_col': 'is_outlier_dbscan',
        'color': '#F59E0B',   # Amber
        'description': "Identifies providers that don't belong to any natural peer cluster — they're outliers in the density structure of the data.",
        'icon': '🔵'
    },
    'benford': {
        'label': "Benford's Law",
        'flag_col': 'is_outlier_benford',
        'color': '#EF4444',   # Red
        'description': "Detects providers whose billing amounts' leading digit distribution deviates from Benford's natural law.",
        'icon': '📐'
    },
    'all': {
        'label': 'All Algorithms (2+ flags)',
        'flag_col': None,
        'color': '#8B5CF6',   # Purple
        'description': 'Shows providers flagged by 2 or more algorithms — the highest confidence fraud candidates.',
        'icon': '🚨'
    }
}

RISK_COLORS = {
    'HIGH':   '#EF4444',  # Red
    'MEDIUM': '#F59E0B',  # Amber
    'LOW':    '#3B82F6',  # Blue
    'NONE':   '#D1D5DB',  # Gray
}

def filter_by_algorithm(df, algo_key):
    """Filter provider_df to only flagged providers for a given algorithm."""
    if algo_key == 'all':
        return df[df['total_flags'] >= 2]
    col = ALGORITHMS[algo_key]['flag_col']
    if col and col in df.columns:
        return df[df[col] == 1]
    return df

def format_currency(val):
    """Format a number as a dollar amount string."""
    if pd.isna(val):
        return 'N/A'
    if val >= 1_000_000:
        return f'${val/1_000_000:.1f}M'
    elif val >= 1_000:
        return f'${val/1_000:.1f}K'
    return f'${val:.2f}'

# =============================================================
# STEP 4: BUILD THE DASH LAYOUT
# =============================================================
# Dash layouts are built using Python objects that map to HTML elements.
# dbc.* components come from dash-bootstrap-components — they give us
# Bootstrap styling (grid system, cards, buttons, etc.) without writing CSS.

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],  # Bootstrap Flatly theme — clean and professional
    title='Virginia Medicaid Fraud Detection'
)

# --- Header ---
# --- Header ---
header = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.I(className='fas fa-shield-alt'), width='auto'),
                dbc.Col(dbc.NavbarBrand('Virginia Medicaid Fraud Detection Dashboard', className='ms-2')),
            ], align='center'),
        ),
        html.Span(
            f'HHS/DOGE Data Release | {len(provider_df):,} Providers | 2018–2024',
            className='text-white-50 navbar-text ms-auto'
        )
    ], fluid=True),
    color='dark', dark=True, className='mb-4'
)


# --- Summary KPI Cards ---
# These 4 cards show top-level statistics at a glance
def make_kpi_card(title, value, subtitle, color):
    """Creates a styled summary card with a big number."""
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className='text-muted mb-1', style={'font-size': '0.8rem', 'text-transform': 'uppercase'}),
            html.H3(value, style={'color': color, 'font-weight': 'bold'}),
            html.Small(subtitle, className='text-muted'),
        ])
    ], className='shadow-sm h-100')

total_paid = provider_df['total_paid_amount'].sum()
high_risk_count = (provider_df.get('risk_tier', pd.Series()) == 'HIGH').sum()
medium_risk_count = (provider_df.get('risk_tier', pd.Series()) == 'MEDIUM').sum()
multi_flagged = (provider_df['total_flags'] >= 2).sum()

kpi_row = dbc.Row([
    dbc.Col(make_kpi_card(
        'Total Virginia Providers', f'{len(provider_df):,}', 'In the dataset', '#1D4ED8'), md=3),
    dbc.Col(make_kpi_card(
        'Total Medicaid Paid', format_currency(total_paid), '2018–2024', '#065F46'), md=3),
    dbc.Col(make_kpi_card(
        'High Risk Providers', f'{high_risk_count:,}', 'Flagged by 3+ algorithms', '#DC2626'), md=3),
    dbc.Col(make_kpi_card(
        'Multi-Algorithm Flags', f'{multi_flagged:,}', 'Flagged by 2+ algorithms', '#7C3AED'), md=3),
], className='mb-4 g-3')

# --- Algorithm Toggle Buttons ---
# These RadioItems let the user switch between algorithms
# dbc.RadioItems renders as Bootstrap toggle buttons when we set inputClassName
algo_selector = dbc.Card([
    dbc.CardHeader(html.H6('🔍 Select Fraud Detection Algorithm', className='mb-0')),
    dbc.CardBody([
        html.P(
            'Toggle between algorithms to see which providers were flagged by each method. '
            'The "All Algorithms" view shows the highest-confidence fraud candidates.',
            className='text-muted mb-3', style={'font-size': '0.85rem'}
        ),
        dbc.RadioItems(
            id='algo-selector',
            options=[
                {
                    'label': html.Span([
                        html.Span(meta['icon'], className='me-1'),
                        html.Strong(meta['label'])
                    ]),
                    'value': key
                }
                for key, meta in ALGORITHMS.items()
            ],
            value='all',   # Default: show all algorithms view
            # inline=True puts all buttons on one row
            inline=True,
            labelClassName='btn btn-outline-secondary me-2 mb-2',
            labelCheckedClassName='active btn-secondary',
            inputClassName='d-none',  # Hide the actual radio input, use button styling
        ),
        # Algorithm description — updates based on selection
        dbc.Alert(id='algo-description', color='light', className='mt-3 mb-0 py-2'),
    ])
], className='shadow-sm mb-4')

# --- Main Charts Row ---
charts_row = dbc.Row([
    # Left: Scatter plot — Total Paid vs Avg Per Claim
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H6('💰 Total Paid vs. Avg Payment per Claim', className='mb-0')),
            dbc.CardBody([
                dcc.Graph(id='scatter-chart', style={'height': '400px'}),
            ])
        ], className='shadow-sm')
    ], md=7),

    # Right: Top flagged providers bar chart
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H6('🏆 Top 15 Flagged Providers by Total Paid', className='mb-0')),
            dbc.CardBody([
                dcc.Graph(id='top-providers-chart', style={'height': '400px'}),
            ])
        ], className='shadow-sm')
    ], md=5),
], className='mb-4 g-3')

# --- Algorithm Comparison Row ---
comparison_row = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H6('📊 Algorithm Comparison — Flags by Method', className='mb-0')),
            dbc.CardBody([
                dcc.Graph(id='algo-comparison-chart', style={'height': '300px'}),
            ])
        ], className='shadow-sm')
    ], md=6),

    dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H6('🎯 Risk Tier Distribution', className='mb-0')),
            dbc.CardBody([
                dcc.Graph(id='risk-tier-chart', style={'height': '300px'}),
            ])
        ], className='shadow-sm')
    ], md=6),
], className='mb-4 g-3')

# --- Benford's Law Visualization ---
benford_row = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H6("📐 Benford's Law — Observed vs. Expected Leading Digit Distribution", className='mb-0')),
            dbc.CardBody([
                dcc.Graph(id='benford-chart', style={'height': '300px'}),
            ])
        ], className='shadow-sm')
    ], md=12),
], className='mb-4')

# --- Data Table ---
table_section = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Col(html.H6('🗂️ Flagged Providers Detail', className='mb-0'), md=6),
            dbc.Col([
                dbc.Input(
                    id='table-search',
                    placeholder='Search by NPI or provider name...',
                    type='text',
                    size='sm'
                )
            ], md=4),
            dbc.Col([
                dbc.Select(
                    id='risk-filter',
                    options=[
                        {'label': 'All Risk Tiers', 'value': 'ALL'},
                        {'label': '🔴 High Risk Only', 'value': 'HIGH'},
                        {'label': '🟡 Medium Risk Only', 'value': 'MEDIUM'},
                        {'label': '🔵 Low Risk Only', 'value': 'LOW'},
                    ],
                    value='ALL',
                    size='sm'
                )
            ], md=2),
        ], align='center')
    ]),
    dbc.CardBody([
        html.Div(id='flagged-count', className='text-muted mb-2', style={'font-size': '0.85rem'}),
        dash_table.DataTable(
            id='providers-table',
            # Columns defined here — data filled in by callback
            columns=[
                {'name': 'NPI',            'id': 'billing_provider_npi'},
                {'name': 'Provider Name',  'id': 'provider_name'},
                {'name': 'City',           'id': 'provider_city'},
                {'name': 'Total Paid',     'id': 'total_paid_amount_fmt'},
                {'name': 'Total Claims',   'id': 'total_claims'},
                {'name': 'Avg/Claim',      'id': 'avg_payment_per_claim_fmt'},
                {'name': 'Flags',          'id': 'total_flags'},
                {'name': 'Risk Tier',      'id': 'risk_tier'},
                {'name': 'Z-Score',        'id': 'is_outlier_zscore'},
                {'name': 'Iso. Forest',    'id': 'is_outlier_isolation_forest'},
                {'name': 'DBSCAN',         'id': 'is_outlier_dbscan'},
                {"name": "Benford's",      'id': 'is_outlier_benford'},
            ],
            page_size=20,
            sort_action='native',      # Allow sorting by clicking column headers
            filter_action='native',    # Allow filtering within the table
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#1E293B',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '12px',
            },
            style_cell={
                'fontSize': '12px',
                'padding': '8px',
                'textAlign': 'left',
                'maxWidth': '200px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            # Color rows based on risk tier
            style_data_conditional=[
                {'if': {'filter_query': '{risk_tier} = "HIGH"'},   'backgroundColor': '#FEF2F2', 'color': '#DC2626'},
                {'if': {'filter_query': '{risk_tier} = "MEDIUM"'}, 'backgroundColor': '#FFFBEB', 'color': '#D97706'},
                {'if': {'filter_query': '{risk_tier} = "LOW"'},    'backgroundColor': '#EFF6FF', 'color': '#2563EB'},
                {'if': {'row_index': 'odd'},                        'backgroundColor': '#F8FAFC'},
            ],
        )
    ])
], className='shadow-sm mb-4')

# --- Assemble full layout ---
# dbc.Container wraps everything in a responsive Bootstrap container
app.layout = dbc.Container([
    header,
    dbc.Container([
        kpi_row,
        algo_selector,
        charts_row,
        comparison_row,
        benford_row,
        table_section,
        # Footer
        html.Hr(),
        html.P([
            '⚠️ ',
            html.Strong('Disclaimer: '),
            'Flagged providers are statistical anomalies, not confirmed fraud cases. '
            'This tool is for research and educational purposes only. '
            'No conclusions should be drawn about any specific provider without further investigation.'
        ], className='text-muted text-center', style={'font-size': '0.8rem'}),
    ], fluid=True)
], fluid=True, className='p-0')

# =============================================================
# STEP 5: CALLBACKS — THE INTERACTIVE LOGIC
# =============================================================
# Callbacks are what make Dash interactive.
# Each callback is a Python function decorated with @callback.
#
# @callback(Output(...), Input(...)) means:
#   "When the Input component changes, run this function
#    and update the Output component with the return value."
#
# This is Dash's core concept: when something changes (user clicks a button,
# types in a box), Dash calls your Python function and re-renders the output.

@callback(
    Output('algo-description', 'children'),
    Input('algo-selector', 'value')
)
def update_algo_description(algo_key):
    """Update the description text when a different algorithm is selected."""
    meta = ALGORITHMS.get(algo_key, {})
    return [
        html.Strong(f"{meta.get('icon', '')} {meta.get('label', '')}  "),
        meta.get('description', '')
    ]


@callback(
    Output('scatter-chart', 'figure'),
    Input('algo-selector', 'value')
)
def update_scatter(algo_key):
    """
    Scatter plot: Total Paid Amount (X) vs Avg Payment per Claim (Y).
    Flagged providers are highlighted in red; normal providers in blue.
    """
    filtered = filter_by_algorithm(provider_df, algo_key)
    flagged_npis = set(filtered['billing_provider_npi'])

    plot_df = provider_df.copy()
    plot_df['flagged'] = plot_df['billing_provider_npi'].isin(flagged_npis)
    plot_df['color_label'] = plot_df['flagged'].map({True: '🚨 Flagged', False: '✅ Normal'})
    plot_df['hover_name'] = plot_df.get('provider_name', plot_df['billing_provider_npi'])
    plot_df['total_paid_fmt'] = plot_df['total_paid_amount'].apply(format_currency)
    plot_df['avg_claim_fmt'] = plot_df['avg_payment_per_claim'].apply(format_currency)

    fig = px.scatter(
        plot_df,
        x='total_paid_amount',
        y='avg_payment_per_claim',
        color='color_label',
        color_discrete_map={'🚨 Flagged': '#EF4444', '✅ Normal': '#93C5FD'},
        hover_name='hover_name',
        hover_data={
            'billing_provider_npi': True,
            'total_paid_fmt': True,
            'avg_claim_fmt': True,
            'total_flags': True,
            'color_label': False,
            'total_paid_amount': False,
            'avg_payment_per_claim': False,
        },
        labels={
            'total_paid_amount': 'Total Paid Amount ($)',
            'avg_payment_per_claim': 'Avg Payment per Claim ($)',
            'color_label': 'Status'
        },
        log_x=True,
        log_y=True,
        opacity=0.7,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='white',
    )
    fig.update_traces(marker=dict(size=6))
    return fig


@callback(
    Output('top-providers-chart', 'figure'),
    Input('algo-selector', 'value')
)
def update_top_providers(algo_key):
    """Horizontal bar chart of top 15 flagged providers by total paid amount."""
    filtered = filter_by_algorithm(provider_df, algo_key)

    if len(filtered) == 0:
        return go.Figure().add_annotation(text='No providers flagged', showarrow=False)

    top15 = filtered.nlargest(15, 'total_paid_amount').copy()
    top15['label'] = top15.get('provider_name', top15['billing_provider_npi']).fillna(top15['billing_provider_npi'])
    top15['label'] = top15['label'].str[:35]   # Truncate long names
    top15['total_paid_fmt'] = top15['total_paid_amount'].apply(format_currency)

    # Color bars by risk tier
    top15['bar_color'] = top15.get('risk_tier', 'LOW').map(RISK_COLORS).fillna('#3B82F6')

    fig = go.Figure(go.Bar(
        x=top15['total_paid_amount'],
        y=top15['label'],
        orientation='h',
        marker_color=top15['bar_color'],
        text=top15['total_paid_fmt'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Total Paid: %{text}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title='Total Paid ($)',
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(tickformat='$,.0f')
    )
    return fig


@callback(
    Output('algo-comparison-chart', 'figure'),
    Input('algo-selector', 'value')
)
def update_algo_comparison(_):
    """Bar chart comparing how many providers each algorithm flagged."""
    algo_keys = ['zscore', 'isolation_forest', 'dbscan', 'benford']
    counts = []
    labels = []
    colors = []

    for key in algo_keys:
        col = ALGORITHMS[key]['flag_col']
        if col and col in provider_df.columns:
            counts.append(int(provider_df[col].sum()))
            labels.append(ALGORITHMS[key]['label'])
            colors.append(ALGORITHMS[key]['color'])

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker_color=colors,
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Flagged: %{y:,}<extra></extra>'
    ))
    fig.update_layout(
        yaxis_title='Providers Flagged',
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='white',
        showlegend=False,
    )
    return fig


@callback(
    Output('risk-tier-chart', 'figure'),
    Input('algo-selector', 'value')
)
def update_risk_tier(_):
    """Pie chart showing distribution of risk tiers across all providers."""
    if 'risk_tier' not in provider_df.columns:
        return go.Figure()

    counts = provider_df['risk_tier'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [RISK_COLORS.get(t, '#D1D5DB') for t in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,   # Donut chart
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        paper_bgcolor='white',
    )
    return fig


@callback(
    Output('benford-chart', 'figure'),
    Input('algo-selector', 'value')
)
def update_benford(_):
    """
    Bar chart comparing observed vs. expected Benford's Law leading digit distribution
    across all flagged providers.
    """
    digits = list(range(1, 10))
    benford_expected = np.log10(1 + 1/np.array(digits))
    benford_expected = benford_expected / benford_expected.sum()

    fig = go.Figure()

    # Expected Benford's distribution
    fig.add_trace(go.Bar(
        x=digits,
        y=benford_expected,
        name="Benford's Expected",
        marker_color='#93C5FD',
        opacity=0.8,
    ))

    # If we have claims data in memory, show observed distribution
    # Otherwise show a placeholder message
    fig.add_annotation(
        text="ℹ️ Run notebook to see observed distribution overlaid here",
        xref='paper', yref='paper',
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=12, color='gray'),
        visible=('leading_digit' not in str(provider_df.columns))
    )

    fig.update_layout(
        xaxis=dict(title='Leading Digit', tickvals=digits),
        yaxis=dict(title='Proportion', tickformat='.1%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='white',
        barmode='group',
    )
    return fig


@callback(
    Output('providers-table', 'data'),
    Output('flagged-count', 'children'),
    Input('algo-selector', 'value'),
    Input('table-search', 'value'),
    Input('risk-filter', 'value'),
)
def update_table(algo_key, search_text, risk_filter):
    """
    Update the data table based on:
    - Which algorithm is selected (algo_key)
    - Search text (NPI or provider name)
    - Risk tier filter
    
    This callback has 3 Inputs — Dash re-runs it whenever ANY of them change.
    """
    # Start with the algorithm-filtered dataset
    filtered = filter_by_algorithm(provider_df, algo_key).copy()

    # Apply risk tier filter
    if risk_filter and risk_filter != 'ALL' and 'risk_tier' in filtered.columns:
        filtered = filtered[filtered['risk_tier'] == risk_filter]

    # Apply text search (case-insensitive, searches NPI and provider name)
    if search_text and search_text.strip():
        search_lower = search_text.strip().lower()
        mask = filtered['billing_provider_npi'].astype(str).str.lower().str.contains(search_lower, na=False)
        if 'provider_name' in filtered.columns:
            mask |= filtered['provider_name'].astype(str).str.lower().str.contains(search_lower, na=False)
        filtered = filtered[mask]

    # Format currency columns for display
    filtered['total_paid_amount_fmt']    = filtered['total_paid_amount'].apply(format_currency)
    filtered['avg_payment_per_claim_fmt'] = filtered['avg_payment_per_claim'].apply(format_currency)

    # Sort by total flags descending, then total paid
    sort_cols = [c for c in ['total_flags', 'total_paid_amount'] if c in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, ascending=False)

    count_text = f'Showing {len(filtered):,} providers'
    if algo_key != 'all':
        count_text += f' flagged by {ALGORITHMS[algo_key]["label"]}'

    return filtered.to_dict('records'), count_text


# =============================================================
# STEP 6: RUN THE APP
# =============================================================

if __name__ == '__main__':
    print()
    print('=' * 55)
    print('🏥 Virginia Medicaid Fraud Detection Dashboard')
    print('=' * 55)
    print(f'   Providers loaded: {len(provider_df):,}')
    print(f'   Fraud flags loaded: {len(flags_df):,}')
    print()
    print('   Opening dashboard at: http://127.0.0.1:8050')
    print('   Press Ctrl+C to stop the server')
    print('=' * 55)
    print()

    # debug=True shows detailed error messages in the browser
    # Set to False before deploying anywhere publicly
    app.run(debug=True, host='127.0.0.1', port=8050)
