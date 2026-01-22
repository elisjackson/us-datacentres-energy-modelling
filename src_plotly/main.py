from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import json

external_stylesheets = [dbc.themes.DARKLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

@app.callback(
    Output("map", "figure"),
    Input("radioitems-input", "value"),
    Input('click-data', 'children')
)
def update_map(radio_selection, click_data):
    if click_data:
        click_data = json.loads(click_data)
    return make_choropleth(radio_selection, click_data)

def make_choropleth(radio_selection, click_data):
    
    if click_data:
        point_index = click_data["points"][0]["pointNumber"]
        selected_id = click_data["points"][0]["location"]
        print(point_index)
        print(selected_id)
    else:
        selected_id = "121-La Pointe-aux-Prairies"

    df = px.data.election()
    geojson = px.data.election_geojson()

    # get a subset of the geojson
    features = geojson["features"]
    highlighted_geojson = {}
    highlighted_geojson["type"] = geojson["type"]
    highlighted_geojson["features"] = []
    for f in features:
        if f["properties"]["district"] == selected_id:
            highlighted_geojson["features"].append(f)

    if radio_selection == "Wind":
        color_on = "Joly"
    elif radio_selection == "PV":
        color_on = "Coderre"

    fig = px.choropleth_map(
        df,
        geojson=geojson,
        locations="district",
        color=color_on,
        featureidkey="properties.district",
        center={"lat": 45.5517, "lon": -73.7073},
        zoom=9,
        range_color=[0, 6500],
        opacity=0.5
    )
    fig.update_layout(margin=dict(r=0, t=0, l=0, b=0))

    # print(geojson)

    fig.add_trace(
        px.choropleth_map(
            df,
            geojson=highlighted_geojson,
            locations="district",
            color=color_on,
            featureidkey="properties.district",
            # center={"lat": 45.5517, "lon": -73.7073},
            # zoom=9,
            range_color=[0, 6500],
            opacity=1
        ).data[0]
    )

    return fig

radioitems = html.Div(
    [
        dbc.Label("Choose one"),
        dbc.RadioItems(
            options=[
                {"label": "Wind", "value": "Wind"},
                {"label": "PV", "value": "PV"},
            ],
            value="Wind",
            id="radioitems-input",
            inline=True,
        ),
    ],
    className="mb-3",
)

accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [   
                                radioitems,
                                dcc.Graph(
                                    id="map",
                                    style={"height": "600px"}
                                )
                            ],
                            width=8,
                        ),
                        dbc.Col(html.Pre(id="click-data"), width=4)
                    ]
                )
            ],
            title="Item 1",
        ),
        dbc.AccordionItem(
            [
                html.P("This is the content of the second section"),
                dbc.Button("Don't click me!", color="danger"),
            ],
            title="Item 2",
        ),
        dbc.AccordionItem(
            "This is the content of the third section",
            title="Item 3",
        ),
    ],
    active_item=0,
)

sidebar = html.Div(
    [
        html.H2("My App", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="#"),
                dbc.NavLink("Analytics", href="#"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

main_content = html.Div(
    [
        html.H1("Hello World"),
        accordion,
    ],
    className="content",
)

app.layout = dbc.Container(
    dbc.Row(
        [
            dbc.Col(sidebar, width=2, className="sidebar-col"),
            dbc.Col(main_content, width=10),
        ],
        className="layout-row g-0",
    ),
    fluid=True,
)

@app.callback(
    Output('click-data', 'children'),
    Input('map', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


if __name__ == "__main__":
    app.run(debug=True)
