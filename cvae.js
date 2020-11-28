// Standard Normal variate using Box-Muller transform.
function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function new_z(zd) {
    for (var i = 0; i < 50; i++) {
       zd[i] = randn_bm()
    }
}

var zd = new Array(82)
for (var i = 0; i < 82; i++) {
    zd[i] = randn_bm()
}

let net;

// load a model from within an asynchronous function
async function loadNet() {
    net = await tf.loadLayersModel('tfjs_g23/model.json');
    // Return a model that outputs an internal activation.
    // const layer = mobilenet.getLayer('conv_pw_13_relu');
    // return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    console.log('Model loaded')
    // return net
}
var predict = function(input) {
    if (net) {
        net.predict(tf.tensor2d(input, [1,82])).array().then(function(output) {
            output = output[0];
            console.log(output[0].toString());
        });
    } else {
        console.log("No net yet...");
        setTimeout(function(){predict(input)}, 100);
    }
}
loadNet();
// predict(zd);

