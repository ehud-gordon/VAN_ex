import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px


def plotly_bar(nums,title=""):
    bins = np.arange(len(nums))
    fig = go.Figure(go.Bar(x=bins, y=nums))
    fig.data[0].text = nums
    fig.update_traces(textposition='inside', textfont_size=12)
    fig.update_layout(bargap=0, title_text=title, title_x=0.5)
    fig.update_traces(marker_color='blue', marker_line_color='blue', marker_line_width=1)
    plot(fig)

def plotly_hist(nums,title="", density=True):
    max = np.max(nums)
    nums_bins = np.arange(max+2)
    counts, bins = np.histogram(nums, bins=nums_bins, density=density)
    fig = go.Figure(go.Bar(x=bins, y=counts))
    fig.data[0].text = counts
    fig.update_traces(textposition='inside', textfont_size=12)
    fig.update_layout(bargap=0, title_text=title, title_x=0.5)
    fig.update_traces(marker_color='blue', marker_line_color='blue', marker_line_width=1)
    plot(fig)

if __name__=="__main__":
    l = np.random.randint(100, size=13)
    print(l)
    plotly_bar(l, title="asd")