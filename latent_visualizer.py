#!/usr/bin/env python3
"""
Latent Space Visualizer - Standalone Web Application
Run this script and open http://localhost:8050 in your browser
"""

import datetime
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import torch
import torchaudio
import base64
import io
from collections import defaultdict
import json
from stable_audio_tools import get_pretrained_model

# Global variables - SET THESE WITH YOUR ACTUAL DATA
sample_dict = None  # Your sample_dict from the notebook
vae = None  # Your VAE model
sr = 44100  # Sample rate
audio_length = 2 * 44100


# Load sample_dict from JSON
def load_sample_dict_from_json(filepath):
  with open(filepath, 'r') as f:
    json_dict = json.load(f)
  
  sample_dict = {}
  for category, files in json_dict.items():
    sample_dict[category] = {}
    for filename, latent_list in files.items():
      # Convert list back to tensor
      sample_dict[category][filename] = torch.tensor(latent_list)
  
  print(f"Loaded sample_dict from {filepath}")
  return sample_dict


# Example of loading (uncomment to use)
sample_dict = load_sample_dict_from_json("sample_dict.json")

model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
vae = model.pretransform.model  # 전체 모델에서 VAE 부분만 추출
vae.eval()  # 평가 모드로 전환


def set_data(your_sample_dict, your_vae, your_sr=44100):
  """Call this function to set your data before running the app"""
  global sample_dict, vae, sr
  sample_dict = your_sample_dict
  vae = your_vae
  sr = your_sr


def load_data():
  """Load and prepare data"""
  global sample_dict, vae
  
  if sample_dict is None or vae is None:
    print("ERROR: Please call set_data(your_sample_dict, your_vae) before running the app")
    return None, None, None, None
  
  print("Loading and processing data...")

  df = pd.DataFrame(None, columns=['index', 'filename', 'label', 'latent', 'flattened', 'x', 'y', 'z'], index=None)

  for label, files in sample_dict.items():
    for fname, latent in files.items():
      df.loc[len(df)] = [
        len(df),
        fname,
        label,
        latent,
        latent.flatten().cpu().numpy(),
        None,
        None,
        None
      ]

  return df


def reduce(df):
  latents = np.stack(df['flattened'].tolist())
  tsne = TSNE(n_components=3, random_state=42, perplexity=min(10, len(latents)-1))
  latents_3d = tsne.fit_transform(latents)
  df['x'] = latents_3d[:, 0]
  df['y'] = latents_3d[:, 1]
  df['z'] = latents_3d[:, 2]


def create_audio_base64(audio_data, sample_rate=44100):
  """Convert audio numpy array to base64 for web playback"""
  # Ensure audio is in correct format
  audio_data = np.array(audio_data, dtype=np.float32)
  
  # Handle different input shapes
  if audio_data.ndim == 1:
    # (samples,) -> (1, samples) for mono
    audio_data = audio_data[np.newaxis, :]
  elif audio_data.ndim == 2:
    if audio_data.shape[0] > audio_data.shape[1]:
      # (samples, channels) -> (channels, samples)
      audio_data = audio_data.T
    # else: already (channels, samples)
  else:
    raise ValueError(f"Unsupported audio shape: {audio_data.shape}")
  
  # Ensure we have at least 1 channel
  if audio_data.shape[0] == 0:
    raise ValueError("Audio data has 0 channels")
  
  # Clip values to valid range
  audio_data = np.clip(audio_data, -1.0, 1.0)
  
  # Convert to torch tensor with correct shape (channels, samples)
  audio_tensor = torch.from_numpy(audio_data)
  
  # Create WAV in memory
  buffer = io.BytesIO()
  try:
    torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
    buffer.seek(0)
    
    # Encode to base64
    audio_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:audio/wav;base64,{audio_base64}"
  except Exception as e:
    print(f"Error in torchaudio.save: {e}")
    print(f"Audio tensor shape: {audio_tensor.shape}")
    print(f"Audio tensor dtype: {audio_tensor.dtype}")
    raise e
  

def load_uploaded_audio(contents, filename, date):
  content_type, content_string = contents.split(',')

  decoded = base64.b64decode(content_string)
  try:
    if 'wav' in filename:
      y, sr = torchaudio.load(io.BytesIO(decoded))
      if y.ndim == 1:
        y = y.unsqueeze(0)
      if y.shape[0] > 2:
        y = y[:2, :]
      if y.shape[0] == 1:
        y = y.repeat(2, 1)
      if sr != 44100:
        y = torchaudio.transforms.Resample(sr, 44100)(y)
      print(f"Loaded WAV file: {filename} at {sr}Hz")
  except Exception as e:
    print(e)
    return html.Div([
      'There was an error processing this file.'
    ])
  
  if y.shape[1] > audio_length:
    # Truncate if longer than 2 seconds
    y = y[:,  audio_length]
  elif y.shape[1] < audio_length:
    # Zero pad if shorter than 2 seconds
    pad_length = audio_length - y.shape[1]
    y = torch.nn.functional.pad(y, (0, pad_length))

  latent = vae.encode(y.unsqueeze(0))
  audio_src = create_audio_base64(y.numpy(), sr)

  return html.Div([
    html.H5(filename),

    html.Hr(),  # horizontal line

    html.Audio(src=audio_src, controls=True, autoPlay=True),
  ]), latent


def create_app():
  """Create and configure the Dash app"""
  # Load data
  df = load_data()
  reduce(df)
  
  if df is None:
    print("Failed to load data. Exiting.")
    return None
  
  # Initialize Dash app
  app = dash.Dash(__name__)
  
  # Create 3D plot
  fig_3d = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='label',
    hover_name='filename',
    title='Latent Space Visualization - Click points to play audio!'
  )
  fig_3d.update_traces(marker=dict(size=8))  # Slightly larger for easier clicking
  fig_3d.update_layout(
    clickmode='event+select',
    dragmode='select',  # Enable selection by default
    scene=dict(
      dragmode='orbit'  # Allow 3D rotation
    ),
    # Enable selection tools in the toolbar
    modebar=dict(
      add=['select2d', 'lasso2d']
    )
  )
  
  # App layout
  app.layout = html.Div([
    html.H1("Latent Space Visualizer", style={'textAlign': 'center', 'margin': '10px 0'}),
    
    html.Div([
      html.P("🎵 Click any point to play that audio file", style={'fontSize': '16px', 'textAlign': 'center', 'margin': '5px 0'}),
      html.P("🎛️ For interpolation: Click two different points to select them", style={'fontSize': '16px', 'textAlign': 'center', 'margin': '5px 0'}),
      html.P("💡 Each click plays the audio AND adds the point to selection (max 2 points)", style={'fontSize': '14px', 'textAlign': 'center', 'margin': '5px 0', 'fontStyle': 'italic', 'color': '#666'}),
    ], style={'margin': '10px', 'backgroundColor': '#f0f0f0', 'padding': '15px', 'borderRadius': '5px'}),
    
    # 3D Plot - Much larger
    dcc.Graph(
      id='3d-plot',
      figure=fig_3d,
      style={'height': '80vh', 'width': '100%'}  # Reduced from 80vh to 50vh
    ),
    
    # Store for selected points
    dcc.Store(id='selected-points', data=[]),
    dcc.Store(id='selection-mode', data=False),  # Track if we're in selection mode
    
    # Control panels in a more compact layout
    html.Div([
      html.Div([
        dcc.Upload(
          id='upload-data',
          children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
          ]),
          style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
          },
          # Allow multiple files to be uploaded
          multiple=False
        ),
        html.Div(id='output-data-upload')
      ]),

      # Single file playback (triggered by clicks)
      html.Div([
        html.H3("🎵 Single File Playback", style={'margin': '10px 0'}),
        html.Div(id='clicked-file-info', style={'margin': '10px 0', 'fontSize': '14px', 'minHeight': '20px'}),
        html.Div(id='single-audio-player'),
      ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'margin': '1%'}),
      
      # Linear interpolation
      html.Div([
        html.H3("🎛️ Linear Interpolation", style={'margin': '10px 0'}),
        html.Div([
          html.Div(id='selected-files-info', style={'margin': '10px 0', 'fontSize': '14px', 'minHeight': '40px'}),
          html.Button('Clear Selection', id='clear-selection-btn', n_clicks=0, 
                     style={'margin': '5px 0', 'backgroundColor': '#ff6b6b', 'color': 'white', 'border': 'none', 'padding': '6px 12px', 'borderRadius': '4px', 'fontSize': '12px'}),
        ]),
        
        html.Div([
          html.Label("Interpolation (0.0 = File1, 1.0 = File2):", style={'fontSize': '14px', 'margin': '10px 0 5px 0'}),
          dcc.Slider(
            id='interp-slider',
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.5,
            marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},  # Fewer marks for compact display
            tooltip={"placement": "bottom", "always_visible": True}
          ),
        ], style={'margin': '10px 0'}),
        
        html.Button('Play Interpolated', id='play-interp-btn', n_clicks=0,
                   style={'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px', 'margin': '10px 0'}),
        html.Div(id='interp-audio-player'),
      ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'margin': '1%'}),
    ], style={'margin': '20px 0'}),
  ], style={'padding': '10px', 'maxWidth': '100%'})
  
  # Callbacks
  @app.callback(
    [Output('single-audio-player', 'children'),
     Output('clicked-file-info', 'children'),
     Output('selected-points', 'data'),
     Output('selected-files-info', 'children')],
    Input('3d-plot', 'clickData'),
    State('selected-points', 'data')
  )
  def handle_point_click(clickData, current_selection):
    if not clickData or 'points' not in clickData or len(clickData['points']) == 0:
      return "", "", current_selection, "No points selected for interpolation"
    
    try:
      # Get the clicked point
      point = clickData['points'][0]
      
      # Try different ways to get point index
      if 'pointIndex' in point:
        point_index = point['pointIndex']
      elif 'pointNumber' in point:
        point_index = point['pointNumber']
      elif 'curveNumber' in point and 'pointNumber' in point:
        point_index = point['pointNumber']
      else:
        # Fallback: try to match by coordinates
        x, y, z = point.get('x'), point.get('y'), point.get('z')
        if x is not None and y is not None and z is not None:
          # Find closest point
          distances = ((df['x'] - x)**2 + (df['y'] - y)**2 + (df['z'] - z)**2)**0.5
          point_index = distances.idxmin()
        else:
          return html.Div("Could not identify clicked point", style={'color': 'red'}), "", current_selection, "Error identifying point"
      
      filename = df.iloc[point_index]['filename']
      label = df.iloc[point_index]['label']
      
      # Play the audio (single file playback)
      latent = latent = df.iloc[point_index]['latent']
      with torch.no_grad():
        decoded = vae.decode(latent).squeeze().cpu().numpy().squeeze()
      
      audio_src = create_audio_base64(decoded, sr)
      audio_player = html.Audio(src=audio_src, controls=True, autoPlay=True)
      
      info_text = html.Div([
        "🎵 Playing:",
        html.Br(),
        f"{filename} (Category: {label})"
      ])
      
      # Handle selection for interpolation
      new_selection = current_selection.copy() if current_selection else []
      
      # Add point to selection if not already selected
      if point_index not in new_selection:
        new_selection.append(point_index)
        # Keep only last 2 selections
        if len(new_selection) > 2:
          new_selection = new_selection[-2:]
      
      # Generate selection info
      if len(new_selection) == 0:
        selection_info = "Click points to select them for interpolation"
      elif len(new_selection) == 1:
        filename1 = df.iloc[new_selection[0]]['filename']
        label1 = df.iloc[new_selection[0]]['label']
        selection_info = html.Div([
          "🎯 Selected 1 point:",
          html.Br(),
          f"{filename1} ({label1})",
          html.Br(),
          "Click another point for interpolation."
        ])
      else:  # len == 2
        filename1 = df.iloc[new_selection[0]]['filename']
        filename2 = df.iloc[new_selection[1]]['filename']
        label1 = df.iloc[new_selection[0]]['label']
        label2 = df.iloc[new_selection[1]]['label']
        selection_info = html.Div([
          "🎯 Selected 2 points:",
          html.Br(),
          f"{filename1} ({label1})",
          html.Br(),
          "    ↔    ",
          html.Br(),
          f"{filename2} ({label2})",
          html.Br(),
          "Ready for interpolation!"
        ])
      
      return audio_player, info_text, new_selection, selection_info
      
    except Exception as e:
      return html.Div(f"Error: {str(e)}", style={'color': 'red'}), f"Error processing click: {str(e)}", current_selection, "Error processing selection"
  
  @app.callback(
    Output('interp-audio-player', 'children'),
    Input('play-interp-btn', 'n_clicks'),
    [State('selected-points', 'data'),
     State('interp-slider', 'value')]
  )
  def play_interpolated(n_clicks, selected_indices, alpha):
    if n_clicks == 0 or not selected_indices or len(selected_indices) != 2:
      return ""
    
    try:
      # Get the two selected files
      filename1 = df.iloc[selected_indices[0]]['filename']
      filename2 = df.iloc[selected_indices[1]]['filename']
      
      latent1 = df.iloc[selected_indices[0]]['latent']
      latent2 = df.iloc[selected_indices[1]]['latent']
      
      # Linear interpolation
      interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
      
      with torch.no_grad():
        decoded = vae.decode(interpolated_latent).squeeze().cpu().numpy().squeeze()
      
      audio_src = create_audio_base64(decoded, sr)
      info_text = f"🎵 Interpolating: {filename1} ({1-alpha:.2f}) + {filename2} ({alpha:.2f})"
      
      return html.Div([
        html.P(info_text, style={'fontSize': '14px', 'fontWeight': 'bold'}),
        html.Audio(src=audio_src, controls=True, autoPlay=True)
      ])
    except Exception as e:
      return html.Div(f"Error: {str(e)}", style={'color': 'red'})
  
  @app.callback(
    [Output('selected-points', 'data', allow_duplicate=True),
     Output('selected-files-info', 'children', allow_duplicate=True)],
    Input('clear-selection-btn', 'n_clicks'),
    prevent_initial_call=True
  )
  def clear_selection(n_clicks):
    if n_clicks > 0:
      return [], "Click points to select them for interpolation"
    return [], "Click points to select them for interpolation"
  

  @app.callback(
    [Output('output-data-upload', 'children'),
     Output('3d-plot', 'figure')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
  )
  def handle_upload(content, fn, date):
    if content is not None:
      children, latent = load_uploaded_audio(content, fn, date)
      df.loc[len(df)] = [
        len(df),
        fn,
        "User",
        latent,
        latent.flatten().cpu().numpy(),
        None,
        None,
        None
      ]
      reduce(df)
      fig_3d = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='label',
        hover_name='filename',
        title='Latent Space Visualization - Click points to play audio!'
      )
      fig_3d.update_traces(marker=dict(size=8))  # Slightly larger for easier clicking
      fig_3d.update_layout(
        clickmode='event+select',
        dragmode='select',  # Enable selection by default
        scene=dict(
          dragmode='orbit'  # Allow 3D rotation
        ),
        modebar=dict(
          add=['select2d', 'lasso2d']
        )
      )
      return children, fig_3d
    return html.Div([
      'Drag and Drop or Select a WAV file to upload and visualize its latent space.'
    ]), dash.no_update
  
  return app

def run_app():
  """Run the Dash application"""
  app = create_app()
  if app is None:
    return
  
  print("Starting Latent Space Visualizer...")
  print("Open http://localhost:8050 in your browser")
  print("Press Ctrl+C to stop the server")
  
  try:
    app.run(debug=True, host='0.0.0.0', port=8050)
  except KeyboardInterrupt:
    print("\nServer stopped.")

if __name__ == '__main__':
  print("Latent Space Visualizer")
  print("=" * 50)
  print("IMPORTANT: Before running, you need to set your data:")
  print("1. Import this module: from latent_visualizer import set_data, run_app")
  print("2. Set your data: set_data(your_sample_dict, your_vae, your_sr)")
  print("3. Run the app: run_app()")
  print("=" * 50)
  
  # If running directly, check if data is set
  if sample_dict is None or vae is None:
    print("\nERROR: No data set. Please use the module as follows:")
    print("\n# In your Python script or notebook:")
    print("from latent_visualizer import set_data, run_app")
    print("set_data(sample_dict, vae, sr)  # your actual variables")
    print("run_app()")
  else:
    run_app() 