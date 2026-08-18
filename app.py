"""Alpha dashboard entrypoint.

    streamlit run app.py
"""

from alpha.dashboard.app import _in_streamlit, main

main()
if not _in_streamlit():
    print("Launch with:  streamlit run app.py")
