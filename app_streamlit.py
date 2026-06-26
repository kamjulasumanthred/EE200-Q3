import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from app import SONG_DB, match_query, process_batch, MIN_MATCHING_HASHES

st.set_page_config(page_title="Zapptain America", layout="wide")

st.title("🎶 Zapptain America — Audio Fingerprinting App")
st.markdown("Identify songs using frequency-domain fingerprints and offset histograms.")

# Index status
if isinstance(SONG_DB, dict) and 'songs' in SONG_DB:
    num_songs = len(SONG_DB['songs'])
else:
    num_songs = len(set(song for list_val in SONG_DB.values() for song, _ in list_val))
st.info(f"**Database status:** Successfully indexed {num_songs} songs.")

tabs = st.tabs(["Single-Clip Mode", "Batch Mode"])

# -----------------------------------------------------------------------------
# Single-Clip Mode
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown("Upload a single query clip to identify the song and visualize the intermediate DSP steps.")
    
    audio_file = st.file_uploader("Upload Query Clip", type=["wav", "mp3", "m4a", "ogg"])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("Identify Song"):
            # Save uploaded file temporarily to run matching in a safe temp dir
            suffix = os.path.splitext(audio_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(audio_file.getbuffer())
                temp_path = temp_file.name
            
            with st.spinner("Analyzing audio clip..."):
                song, score, fig1, fig2, fig3 = match_query(temp_path, db=SONG_DB, use_pairs=True, generate_plots=True)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            
            # Show results
            if song != "No Match Found":
                st.success(f"**Predicted Song:** {song} (aligned votes = {score})")
            else:
                st.warning(f"No confident match found (best score {score} < threshold {MIN_MATCHING_HASHES})")
                
            # Display plots
            if fig1 and fig2 and fig3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.pyplot(fig1)
                with col2:
                    st.pyplot(fig2)
                with col3:
                    st.pyplot(fig3)
                
                # Close figures to save memory
                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig3)

# -----------------------------------------------------------------------------
# Batch Mode
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("Upload multiple query clips to generate the required `results.csv` file.")
    
    batch_files = st.file_uploader("Upload Query Clips", type=["wav", "mp3", "m4a", "ogg"], accept_multiple_files=True)
    
    if batch_files:
        if st.button("Process Batch"):
            temp_paths = []
            
            with st.spinner("Processing batch..."):
                for uploaded_file in batch_files:
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name
                    
                    # Attach orig_name attribute so process_batch uses it
                    class FileWrapper:
                        def __init__(self, path, orig_name):
                            self.path = path
                            self.orig_name = orig_name
                    
                    temp_paths.append(FileWrapper(temp_path, uploaded_file.name))
                
                csv_path = process_batch(temp_paths)
                
                # Clean up temp files
                for f_wrap in temp_paths:
                    if os.path.exists(f_wrap.path):
                        os.remove(f_wrap.path)
            
            # Show results and download button
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                st.success("Batch processing complete!")
                st.dataframe(df)
                
                with open(csv_path, "r") as f:
                    csv_data = f.read()
                
                st.download_button(
                    label="Download results.csv",
                    data=csv_data,
                    file_name="results.csv",
                    mime="text/csv"
                )
                
                # Clean up results.csv
                os.remove(csv_path)
