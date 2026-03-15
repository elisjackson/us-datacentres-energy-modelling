"""Root entrypoint for Plotly Cloud publishing and local runs."""

from src.main import app as imported_app, server

app = imported_app

if __name__ == "__main__":
    app.run(debug=True)
