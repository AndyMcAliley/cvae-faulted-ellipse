'''
https://panel.holoviz.org/user_guide/Links.html#Linking-using-custom-JS-code
/Users/andymcaliley/Codes/DoZen/scripts/timeline.py
https://www.tensorflow.org/js/tutorials/conversion/import_keras
https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/about-github-pages
'''

import numpy as np
import bokeh
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearColorMapper, ColumnDataSource, CustomJS, Slider, ColorBar
from bokeh.palettes import viridis
# import panel

output_file("db.html")

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

m = np.arange(1024).reshape(32,32)/1024*6-3

# plot using above function
# mp = plot_model(m)

# plot outright using ColumnDataSource
source = ColumnDataSource(data=dict(image=[m]))
p = figure()
p.x_range.range_padding = p.y_range.range_padding = 0
color_mapper = LinearColorMapper(viridis(256), low=-1, high=1)
p.image(image='image', x=0, y=-800, dw=800, dh=800, color_mapper=color_mapper,
        source=source)
p.grid.grid_line_color = None
color_bar = ColorBar(color_mapper=color_mapper)
p.add_layout(color_bar, 'right')

jscode = '''
var data = source.data;
var f = cb_obj.value;
zd[48] = f;

var predict = function(input) {
    if (net) {
        net.predict(tf.tensor2d(input, [1,82])).array().then(function(output) {
            output = output[0];
            console.log(output[0].toString());
            var m = data['image'];
            console.log(m[0].length.toString())
            console.log(output.length.toString())
            var ni = output.length;
            // Assume all output[i] are of equal length
            var nj = output[0].length;
            var ij = 0;
            var im = 0;
            for (var i = 0; i < ni; i++) {
                for (var j = 0; j < nj; j++){
                    ij = i*output[i].length + j;
                    im = ni*nj - ij - 1
                    m[0][im] = output[i][j];
                }
            }
            // for (var i = 0; i < m[0].length; i++) {
                // m[0][i] = output[i];
            // }
            // m[0] = output[0];
            // m[1] = output[1];
            source.change.emit();
        });
    } else {
        console.log("No net yet...");
        setTimeout(function(){predict(input)}, 100);
    }
}
predict(zd);
'''
# m=output; is most iffy


# pure bokeh
slider_change = CustomJS(
    args=dict(source=source),
    code=jscode
    # code=""" 
    # var data = source.data;
    # var m = data['image']
    # var f = cb_obj.value;
    # m[0][0] = f
    # source.change.emit();
    # """)
    )

'''
m is an object of length 1
m[0] is of length 1024
Here are some log commands to verify:
    console.log(typeof(m))
    console.log(m.length.toString());
    console.log(m[0].length.toString());
setting an element of m directly seems to work!
The plot is updated accordingly.
'''

n_sliders = 10
sliders = []
for i_slider in range(n_sliders):
    slider = Slider(start=-3, end=3, value=0, step=0.1, title=f'z{i_slider}')
    # pure bokeh
    slider_change = CustomJS(
        args=dict(source=source, iz=i_slider),
        code=jscode
        )

    slider.js_on_change('value', slider_change)
    sliders.append(slider)

slider1 = Slider(start=-3, end=3, value=0, step=0.1, title='z1')
slider1.js_on_change('value', slider_change)
layout = row(p,slider1)
show(layout)

# use panel
# db = panel.Column(mp).servable()
# db = panel.Column(mp)
# db.save('dashboard.html',embed=True)
