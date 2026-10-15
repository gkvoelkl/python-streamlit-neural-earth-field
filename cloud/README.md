# Online version

This folder is only for running the app on [Streamlit Community Cloud](https://streamlit.io/cloud).
Local installs don't use it.

- `streamlit_app.py` is the entry point to choose in Community Cloud. It switches the app to
  its online mode (`NEURAL_EARTH_CLOUD=1`): no download prompt, 1 000 training steps by
  default and a note that training runs on a shared CPU.
- `requirements.txt` installs PyTorch for the CPU only. Community Cloud reads the dependency
  file next to the entry point first, so it takes this one instead of `uv.lock`.
- The elevation grids, already averaged down from ETOPO 2022 (NOAA NCEI,
  DOI [10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74)), come from the GitHub
  release [`data-v1`](https://github.com/gkvoelkl/python-streamlit-neural-earth-field/releases/tag/data-v1)
  and are downloaded on first use.

Try the online mode locally in a fresh environment:

```bash
uv venv /tmp/neural-earth-cloud --python 3.14
uv pip install --python /tmp/neural-earth-cloud -r cloud/requirements.txt
NEURAL_EARTH_DATA_DIR=/tmp/neural-earth-data /tmp/neural-earth-cloud/bin/streamlit run cloud/streamlit_app.py
```
