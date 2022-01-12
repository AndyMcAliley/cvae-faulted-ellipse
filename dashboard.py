'''
https://panel.holoviz.org/user_guide/Links.html#Linking-using-custom-JS-code
/Users/andymcaliley/Codes/DoZen/scripts/timeline.py
https://www.tensorflow.org/js/tutorials/conversion/import_keras
https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/about-github-pages
'''

import numpy as np
import bokeh
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import LinearColorMapper, ColumnDataSource, CustomJS, Slider, ColorBar
from bokeh.palettes import viridis
# import panel

out_html = "db.html"
output_file(out_html, title="CVAE")

"""
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
"""

# data locations
data_locations = np.linspace(200, 600, num=32)
# define true data
true_data = np.array([
    0., 0.01080575, 0.02209415, 0.03374726, 0.04563441, 0.05762168, 0.06958213,
    0.08140488, 0.09300181, 0.10431098, 0.11529678, 0.12594756, 0.13627152,
    0.14629198, 0.15604261, 0.16556319, 0.1748961, 0.18408361, 0.19316588,
    0.20217957, 0.21115706, 0.22012593, 0.22910882, 0.23812347, 0.24718296,
    0.25629598, 0.26546725, 0.27469787, 0.28398584, 0.29332641, 0.30271258,
    0.31213546
])
d0 = np.array([
    0., 0.01118805, 0.02256263, 0.03406886, 0.04564909, 0.05724883, 0.0688213,
    0.08032905, 0.0917424, 0.10303667, 0.11419064, 0.12518763, 0.13601938,
    0.14669007, 0.15721782, 0.16763123, 0.17796191, 0.18823761, 0.19847761,
    0.20868922, 0.21886515, 0.22898376, 0.23901328, 0.24891957, 0.2586741,
    0.26825819, 0.27766238, 0.28688262, 0.29591509, 0.30475159, 0.31337732,
    0.32177198
])

# load models
img_rows = 32
img_cols = 32
# m = (np.arange(1024).reshape(img_rows,img_cols)/1024 - 0.5)*2
m = np.loadtxt('m0.txt').reshape(img_rows, img_cols)
m = np.flipud(m)

# plot geometry
model_height = 500
model_width = model_height
data_height = 300
data_width = model_width
colorbar_vertial_pad = 10
colorbar_plot_height = model_height - 34 - colorbar_vertial_pad
colorbar_height = colorbar_plot_height - 48 - colorbar_vertial_pad
colorbar_width = 100

# plot outright using ColumnDataSource
# source = ColumnDataSource(data=dict(image=[m]))
# p = figure()
# p.x_range.range_padding = p.y_range.range_padding = 0
# color_mapper = LinearColorMapper(viridis(256), low=-1, high=1)
# p.image(image='image', x=0, y=-800, dw=800, dh=800, color_mapper=color_mapper,
#         source=source)
# p.grid.grid_line_color = None
# color_bar = ColorBar(color_mapper=color_mapper)
# p.add_layout(color_bar, 'right')




# plot outright using ColumnDataSource
cds_model = ColumnDataSource(data=dict(image=[m]))
pm = figure(title='Density model', height=model_height, width=model_width)
pm.title.text_font_size = "20px"
pm.x_range.range_padding = pm.y_range.range_padding = 0
color_mapper = LinearColorMapper(palette=viridis(256), low=-1, high=1)
pm.image(image='image', x=0, y=-800, dw=800, dh=800, color_mapper=color_mapper,
        source=cds_model)
pm.grid.grid_line_color = None
pm.xaxis.axis_label = 'Distance, m'
pm.yaxis.axis_label = 'Elevation, m'
# pm.xaxis.ticker = np.linspace(0, 800, num=800//100 + 1)
cb = ColorBar(color_mapper=color_mapper, height=colorbar_height, location=(0,0))
# pm.add_layout(cb, 'right')
pmcb = figure(title='density, g/cm³', title_location='right',
              height=colorbar_plot_height, width=colorbar_width, 
              toolbar_location=None, min_border=0,
              outline_line_color=None
             )
pmcb.add_layout(cb, 'center')
pmcb.title.align = 'center'
pmcb.title.text_font_size = '16px'

# plot data using ColumnDataSource
cds_data = ColumnDataSource(data=dict(x=data_locations, y_true=true_data, y_pre=d0))
pd = figure(title='Gravity data', height=data_height, width=data_width, x_range=pm.x_range)
pd.title.text_font_size = "20px"
pd.line('x', 'y_true', source=cds_data, legend_label="True data", line_width=3)
pd.circle('x', 'y_true', source=cds_data, legend_label="True data")
pd.line('x', 'y_pre', source=cds_data, legend_label="Modeled data", line_color='orange', line_width=3)
pd.legend.location = "top_left"
pd.xaxis.axis_label = 'Distance, m'
pd.yaxis.axis_label = 'Gravity, mGal'

# Load sensitivity matrix
G = np.load('sensitivity.npy')

jscode = '''
var model = model_source.data;
var y_pre = data_source["data"]["y_pre"];
var f = cb_obj.value;
zd[iz] = f;

var predict = function(input) {
    if (net) {
        net.predict(tf.tensor2d(input, [1,82])).array().then(function(output) {
            output = output[0];
            console.log(output[0].toString());
            var m = model['image'];
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
            model_source.change.emit();
            const nd = y_pre.length;
            // transform model from tanh to density
            // m/2.25 + 2.4
            const m_matrix = math.add(math.divide(math.matrix(Array.from(m[0])), 2.25), 2.4);
            // console.log(math.size(G));
            // console.log(math.size(m_matrix));
            const d_pre = math.multiply(G,m_matrix);
            for (var ii = 0; ii < nd; ii++) {
                y_pre[ii] = math.subset(d_pre, math.index(ii));
            }
            data_source.change.emit();
        });
    } else {
        console.log("No net yet...");
        setTimeout(function(){predict(input)}, 100);
    }
}
predict(zd);
'''


# pure bokeh
iz_sort = [48, 19, 24, 49, 34, 41, 37, 39, 45, 18, 27, 47, 28, 17, 38, 43, 13,
           33, 14, 40, 11, 20, 26, 2, 7, 31, 1, 16, 30, 8, 25, 23, 21, 29, 36,
           5, 22, 0, 10, 6, 4, 32, 35, 3, 46, 42, 44, 12, 15, 9]
iz = iz_sort[0]

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

sliders = []
for ii,iz in enumerate(iz_sort[:10]):
    slider_change = CustomJS(
        args=dict(model_source=cds_model, data_source=cds_data, iz=iz),
        # code=jscode
        code=""" 
        var model = model_source.data;
        var m = model['image']
        var y_pre = data_source["data"]["y_pre"];
        const nd = y_pre.length;
        var f = cb_obj.value;
        m[0][1024-iz-1] = f;
        model_source.change.emit();
        //
        //
        // transform model from tanh to density
        // m/2.25 + 2.4
        const m_matrix = math.add(math.divide(math.matrix(Array.from(m[0])), 2.25), 2.4);
        // console.log(math.size(G));
        // console.log(math.size(m_matrix));
        const d_pre = math.multiply(G,m_matrix);
        for (var ii = 0; ii < nd; ii++) {
            y_pre[ii] = math.subset(d_pre, math.index(ii));
        }
        data_source.change.emit();
        // debugger
        """
        )
    slider1 = Slider(start=-3, end=3, value=0, step=0.1, title='z{}'.format(ii+1))
    slider1.js_on_change('value', slider_change)
    sliders.append(slider1)
layout = row(column(pd,row(pm,pmcb)),column(sliders))
save(layout)

# Add src to html
html_src = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/10.0.2/math.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script> 
        <script src="cvae.js" type="text/javascript"></script>
"""

with open(out_html, "r") as f:
    contents = f.readlines()

contents.insert(18, html_src)

with open(out_html, "w") as f:
    contents = "".join(contents)
    f.write(contents)

# use panel
# db = panel.Column(mp).servable()
# db = panel.Column(mp)
# db.save('dashboard.html',embed=True)
