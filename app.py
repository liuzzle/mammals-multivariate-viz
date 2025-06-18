import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State

df = pd.read_csv('mammals.csv')

# Preprocessing: Filling missing values with KNN
# List of numeric columns to impute, exclude 'species' for KNN
# similar code in exercise 3, but here I use KNN:
numeric_cols = [
    'body_wt', 'brain_wt', 'non_dreaming', 'dreaming', 'total_sleep',
    'life_span', 'gestation', 'predation', 'exposure', 'danger'
]

# Track missing before filling
missing_before = df[numeric_cols].isnull().sum().reset_index()  # reset index to convert to df
missing_before.columns = ['Column', 'MissingBefore']  # rename columns
print("DF before KNN", missing_before) # print for verification

# use KNN to impute missing values
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Normalize each variable: mean 0, std 1
df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

# PCA inspo: https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/
# Based on source, could also use StandardScaler() to standardize the data
# -> would do the same as above: we get similar results, rounding differences
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#print("Normalized DF", df[numeric_cols].head())

# reduce dimensionality, keep 10 PCs
pca = PCA(n_components=10)
pca_result = pca.fit_transform(df[numeric_cols])
explained_var = pca.explained_variance_ratio_

#pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
#pca_df['species'] = df['species'].values

# Add PCA results and metadata
df['index'] = df.index
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(10)])
pca_df['species'] = df['species'].values
pca_df['index'] = df['index'].values  
for col in numeric_cols:
    pca_df[col] = df[col].values

# MDS inspo: https://www.geeksforgeeks.org/multidimensional-scaling-mds-using-scikit-learn/
# Project data to 2D
mds = MDS(n_components=2, random_state=0)
mds_result = mds.fit_transform(df[numeric_cols])

# Add MDS results and metadata
mds_df = pd.DataFrame(data=mds_result, columns=['MDS1', 'MDS2'])
mds_df['species'] = df['species'].values
mds_df['index'] = df['index'].values 
for col in numeric_cols:
    mds_df[col] = df[col].values

# Reset index (for "safety")
pca_df = pca_df.reset_index()    
mds_df = mds_df.reset_index()

# mapping of labels: variable – human-readable
var_labels = {
    'body_wt': 'Body Weight',
    'brain_wt': 'Brain Weight', 
    'non_dreaming': 'Non-Dreaming Sleep',
    'dreaming': 'Dreaming Sleep',
    'total_sleep': 'Total Sleep',
    'life_span': 'Life Span',
    'gestation': 'Gestation',
    'predation': 'Predation',
    'exposure': 'Exposure',
    'danger': 'Danger'
}

# label PC axes with most influential variables
def get_pc_label(pc_idx, loadings, variable_names, num_top_vars=2):
    pc_loadings = np.abs(loadings[pc_idx])
    top_indices = np.argsort(pc_loadings)[-num_top_vars:][::-1]
    top_vars = [var_labels.get(variable_names[i], variable_names[i]) for i in top_indices]
    return f"PC{pc_idx+1}: {' & '.join(top_vars)}"

# add explained variance to axis label
def get_pc_axis_label(pc_idx, explained_var):
    return f"PC{pc_idx+1} ({explained_var[pc_idx]*100:.3f}% variance)"

# dropdown options for PCA components
pc_options = [
    {'label': get_pc_label(i, pca.components_, numeric_cols), 'value': i}
    for i in range(len(explained_var))
]

# Dropdown options for color coding
color_var_options = [
    {'label': var_labels.get(col, col), 'value': col} for col in numeric_cols
]

# Dash App Layout
app = dash.Dash(__name__)
app.title = "Sleep in Mammals Dataset"

app.layout = html.Div([
    html.H1("Sleep in Mammals Dataset"),
    
    # Dropdowns for PCA
    html.Div([
        html.Label("X-axis:", style={'marginLeft': '10px', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='dropdown-x',
            options=pc_options,
            value=0,
            style={'width': '340px', 'display': 'inline-block'}
        ),
        html.Span(id='x-var', style={'marginLeft': '10px', 'fontWeight': 'bold'}),

        html.Label("Y-axis:", style={'marginLeft': '20px', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='dropdown-y',
            options=pc_options,
            value=1,
            style={'width': '340px', 'display': 'inline-block'}
        ),
        html.Span(id='y-var', style={'marginLeft': '10px', 'fontWeight': 'bold'}),
        
        html.Label("Color by:", style={'marginLeft': '2px', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='dropdown-color',
            options=color_var_options,
            value=numeric_cols[0],
            style={'width': '200px', 'display': 'inline-block'}
        )
    ], style={'marginBottom': '20px'}),
    
    # Scatterplots
    html.Div([
        dcc.Graph(id='scatter-pca', style={'display': 'inline-block', 'width': '48%'}),
        dcc.Graph(id='scatter-mds', style={'display': 'inline-block', 'width': '48%'}),
    ]),
    
    # Bar Chart
    dcc.Graph(id='bar-loadings'),

    # Store selected points for linked selection
    dcc.Store(id='selected-points', data=[])
])

# inspo for selection linking: https://dash.plotly.com/interactive-graphing 

# callback to update plots
@app.callback(
    [Output('scatter-pca', 'figure'),
     Output('scatter-mds', 'figure'),
     Output('x-var', 'children'),
     Output('y-var', 'children'),
     Output('bar-loadings', 'figure'),
     Output('selected-points', 'data')],
    [Input('dropdown-x', 'value'),
     Input('dropdown-y', 'value'),
     Input('dropdown-color', 'value'),
     Input('scatter-pca', 'selectedData'),
     Input('scatter-mds', 'selectedData')],
    [State('selected-points', 'data')]
)

# note: i wasn't able to handle the linked selection :( 
# it only shows for a split second where it is in the other plot...

def update_plots(x_idx, y_idx, color_var, selected_pca, selected_mds, stored_selection):

    # extract selected indices from either graph
    def extract_indices(selected_data):
        if selected_data and 'points' in selected_data:
            return set(point['customdata'][0] for point in selected_data['points'])
        return set()

    selected_indices_pca = extract_indices(selected_pca)
    selected_indices_mds = extract_indices(selected_mds)

    # Combine logic: intersection if both have selections, otherwise union
    if selected_indices_pca and selected_indices_mds:
        selected_indices = selected_indices_pca & selected_indices_mds
    else:
        selected_indices = selected_indices_pca | selected_indices_mds

    selected_data = list(selected_indices)

    x_axis_label = get_pc_axis_label(x_idx, explained_var)
    y_axis_label = get_pc_axis_label(y_idx, explained_var)


    # PCA Plot
    fig_pca = px.scatter(
        pca_df,
        x=f'PC{x_idx+1}',
        y=f'PC{y_idx+1}',
        color=color_var,
        color_continuous_scale='RdBu',
        hover_data={col: True for col in ['species'] + numeric_cols},
        custom_data=['index']
    )
    fig_pca.update_traces(
        selectedpoints=selected_data,
        marker=dict(size=12, opacity=0.9),
        unselected=dict(marker=dict(opacity=0.3))
    )

    # MDS Plot
    fig_mds = px.scatter(
        mds_df,
        x='MDS1',
        y='MDS2',
        color=color_var,
        color_continuous_scale='RdBu',
        hover_data={col: True for col in ['species'] + numeric_cols},
        custom_data=['index']
    )
    fig_mds.update_traces(
    selectedpoints=selected_data,
    marker=dict(size=12, opacity=0.9),
    unselected=dict(marker=dict(opacity=0.3))
    )
    fig_mds.update_layout(
        dragmode='select',
        uirevision=True,
        xaxis_title='MDS Dimension 1',
        yaxis_title='MDS Dimension 2',
        title='Multidimensional Scaling'
    )

    # Bar chart for loadings
    loadings = pca.components_
    x_load = loadings[x_idx] * explained_var[x_idx]
    y_load = loadings[y_idx] * explained_var[y_idx]
    
    variables_extended = [var_labels.get(v, v) for v in numeric_cols] * 2
    loadings_values = np.concatenate([x_load, y_load])
    pc_labels = [f'PC{x_idx+1}'] * len(numeric_cols) + [f'PC{y_idx+1}'] * len(numeric_cols)
    
    bar_fig = px.bar(
        x=variables_extended,
        y=loadings_values,
        color=pc_labels,
        barmode='group',
        labels={'x': 'Variable', 'y': 'Loading (scaled by variance)', 'color': 'PC'},
        title='Loadings of Selected Principal Components'
    )

    return fig_pca, fig_mds, "", "", bar_fig, selected_data

if __name__ == '__main__':
    app.run(debug=True, port=8000)