# Scottish Path Grading Tool (LIDAR)
Streamlit web app for path managers.

Upload a GPX → sample elevations from Scottish Public Sector LiDAR DTM → get slopes, suggested grade, and an editable six-part route panel following the Path Managers Guide to Grading.

## Local use (Windows / Mac / Linux)

1. Install Python 3.9+ from python.org
2. Open a terminal / command prompt in this folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

3. The app opens in your browser (usually http://localhost:8501).

## Sharing with other path managers

- **Easiest**: Deploy free on Streamlit Community Cloud (https://streamlit.io/cloud).  
  Create a GitHub repo, push these files, then connect it. Others just open the link.
- Or run it on any machine with Python and share the local address on your network.
- Or package as a desktop app later with PyInstaller if needed.

## Data notes

- Uses Scottish Public Sector LiDAR DTM (Open Government Licence).
- Typical vertical accuracy ±15 cm where coverage exists.
- Coverage is good across much of Scotland but not 100%.
- Always verify the steepest sections in the field with a clinometer.
- First run for a new area downloads tiles (can take a few minutes); subsequent runs are faster if you cache tiles.

## Files

- `app.py` – the Streamlit app
- `requirements.txt` – dependencies
