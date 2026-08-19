# AgriScan — UAV Crop Disease Detection (Streamlit)

A functional demo: upload a field image, and the app computes a real
vegetation index (VARI — an RGB-only NDVI proxy) from the pixels, grades
severity zones relative to that image's own distribution, and logs every
scan to an in-session dashboard with a trend chart.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy for free — Streamlit Community Cloud (fastest, ~3 minutes)

1. **Push this folder to a GitHub repo.**
   - Create a new repo on github.com (public is fine, and required for the free tier).
   - Upload `app.py` and `requirements.txt` to it (drag-and-drop on GitHub's
     web UI works, or `git push` if you're comfortable with git).

2. **Go to** [share.streamlit.io](https://share.streamlit.io) **and sign in with GitHub.**

3. Click **"New app"**, then select:
   - Repository: the repo you just created
   - Branch: `main`
   - Main file path: `app.py`

4. Click **Deploy**. Streamlit installs `requirements.txt` and builds the
   app automatically — first deploy takes 1–2 minutes.

5. You'll get a public URL like `https://your-app-name.streamlit.app` —
   that's what you demo from and put in your submission.

No servers, no Docker, no config needed — Streamlit Community Cloud handles
all of it from just these two files.

## What's actually real vs. simulated

- **Real vegetation index**: VARI is computed live from whatever image you
  upload, using actual pixel math. Severity zones and the affected-area %
  are derived from that computed index, not random numbers.
- **Real disease classification**: the diagnosis comes from
  [`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification) —
  a MobileNetV2 model fine-tuned on the PlantVillage dataset (38 classes
  across common crops). It runs live via Hugging Face `transformers`, and
  the app shows its raw top-5 predictions in an expander for transparency.
- **Note on accuracy**: this model was trained on close-up single-leaf
  photos. It works best on that kind of image — a wide aerial field shot
  will still get graded for severity correctly (that part is pure pixel
  math, not the model), but the disease *label* will be less reliable on
  images unlike its training data. Say this plainly if judges ask; it's
  a stronger answer than overselling it.
- **First run is slower**: the model (~10MB) downloads once from Hugging
  Face the first time you click "Run scan," then stays cached for the
  rest of the session via `st.cache_resource`. Budget ~10–20 seconds for
  that first click during your live demo, or click "Run scan" once
  yourself before presenting to warm the cache.
