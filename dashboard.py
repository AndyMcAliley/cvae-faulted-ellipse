'''
https://panel.holoviz.org/user_guide/Links.html#Linking-using-custom-JS-code
/Users/andymcaliley/Codes/DoZen/scripts/timeline.py
https://www.tensorflow.org/js/tutorials/conversion/import_keras
https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/about-github-pages
'''

import numpy as np
import bokeh
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper
from bokeh.palettes import viridis
import panel


# x = np.arange(32)*25
# y = np.arange(32)*25 - 800

def plot_model(m):
    '''
    Plot a density model
    '''
    p = figure()
    p.x_range.range_padding = p.y_range.range_padding = 0
    color_mapper = LinearColorMapper(viridis(256))
    p.image(image=[m], x=0, y=-800, dw=800, dh=800, color_mapper=color_mapper,
            level="image")
    p.grid.grid_line_color = None
    return p




def update():
    '''
    Call network, update plots
    '''
    return None

m = np.arange(1024).reshape(32,32)
mp = plot_model(m)
db = panel.Column(mp).servable()
# db = panel.Column(mp)
# db.save('dashboard.html',embed=True)
