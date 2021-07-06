import numpy as np
import plotly
############ fig.write_html ############
def plotly_write_html(fig, name="tmp", auto_open=False):
    fig.write_html(f"{name}.html", auto_open=auto_open)

############ plotly.offline.plot ############
def plotly_offline(fig):
    plotly.offline.plot(fig)

############## JSON ###############
def write_json(fig, name):
    fig.write_json(name +'.JSON')
def read_json(path):
    fig = plotly.io.read_json(path)
    return fig
########### EXAMPLES ###########
def frames_example():
    # CREATE fixed scatter
    t = np.linspace(-1, 1, 50)  # [-1,1]
    xm = np.min(t) - 1.5
    xM = np.max(t) + 1.5
    ym = np.min(t) - 1.5
    yM = np.max(t) + 1.5
    fixed_scatter = go.Scatter(x=t, y=t, mode='lines', name="fixed", line=dict(color="blue"))

    # CREATE frames
    slide_scatter = go.Scatter(x=t, y=t, mode='lines', name="slide", line=dict(color="blue"))
    s = np.linspace(-1, 1, 25)  # [-1, 1]
    frames = []
    for k in range(25):
        scat = go.Scatter(x=[s[k]], y=[s[k]], mode="markers", marker=dict(color="red"))
        frame = go.Frame(data=[scat], traces=[1])
        frames.append(frame)

    # CREATE figure
    fig = go.Figure(data=[fixed_scatter, slide_scatter],
                    layout=go.Layout(
                        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
                        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
                        updatemenus=[dict(type="buttons",
                                          buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None])])]),
                    frames=frames
                    )

    fig.update_layout(xaxis_title="X Axis", yaxis_title="Y Axis", title=r"my_title", title_x=0.5)

def slider_example():
    fig = go.Figure()

    # CREATE fixed scatter
    t = np.linspace(-1, 1, 50)  # [-1,1]
    xm = np.min(t) - 1.5
    xM = np.max(t) + 1.5
    ym = np.min(t) - 1.5
    yM = np.max(t) + 1.5
    fig.add_trace(go.Scatter(x=t, y=t, mode='lines', name="fixed", line=dict(color="blue")))

    # Add traces, one for each slider step
    s = np.linspace(-1, 1, 25)  # [-1, 1]
    for k in range(25):
        scat = go.Scatter(x=[s[k]], y=[s[k]], mode="markers", marker=dict(color="red"), name=f's={str(k)}',
                          visible=False)
        fig.add_trace(scat)
    fig.data[1].visible = True

    # Create and add slider
    steps = []
    for i in range(1, len(fig.data)):
        vises = [False] * len(fig.data)
        vises[0] = True; vises[i] = True
        step = dict(
            method="update",
            args=[{"visible": vises},
                  {"title": "Slider switched to step: " + str(i - 1)}],  # layout attribute
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "step: "},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.update_layout(xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
                      yaxis=dict(range=[ym, yM], autorange=False, zeroline=False))

    fig.update_layout(xaxis_title="X Axis", yaxis_title="Y Axis", title=r"my_title", title_x=0.5)

    plotly.offline.plot(fig)

########################################


if __name__=="__main__":
    import plotly.graph_objects as go
    
    scatter1 = go.Scatter(x=[2, 3], y=[2, 3], mode='lines', name="scatter1", line=dict(color="red"), marker=dict(color="green")  )
    scatter2 = go.Scatter(x=[0, 1], y=[0, 2], mode='markers', name="scatter2", marker=dict(color="blue"), line=dict(color="yellow")  )
    fig = go.Figure([scatter1, scatter2])

    fig.update_layout(font=dict(size=18))
    fig.update_layout(title_text="my_title", title_x=0.5)
    fig.update_layout(xaxis_title="X Axis", yaxis_title="Y Axis")
    # fig.update_layout(xaxis_range=[-4,4], yaxis_range=[-4,4])

    # fig = go.Figure(data=go.Scatter(x=[0, 1, 2], y=[5, 10, 15], mode='markers'))
    fig.write_html("tmp.html", auto_open=False)
    # plotly.offline.plot(fig)