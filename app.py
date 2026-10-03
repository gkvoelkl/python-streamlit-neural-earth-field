"""Entry point for ``streamlit run app.py``.

``streamlit run`` executes the target file as a script, not as a package module --
that is why the launcher lives up here and the package is imported normally.
"""

from python_streamlit_neural_earth_field.app import main

main()
