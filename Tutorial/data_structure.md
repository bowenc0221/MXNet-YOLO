# Data structure for darknet

* [Network struct](#Network) : data structure for saving the entire network  
* [Layer struct](#Layer) : data structure for saving each layer  

# Network
## defination
```C
typedef struct network{
    int n;  // number of layers
    int batch;
    size_t *seen; // number of images seen?
    int *t;
    float epoch;
    int subdivisions;
    layer *layers; // array of layers, e.g. layers[0] return first layer
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif

} network;
```
# Layer
## defination
```C
struct layer{  
    LAYER_TYPE type;  // layer type
    ACTIVATION activation;  // activation type
    COST_TYPE cost_type;  
    void (*forward)   (struct layer, struct network);  
    void (*backward)  (struct layer, struct network);  
    void (*update)    (struct layer, update_args);  
    void (*forward_gpu)   (struct layer, struct network);  
    void (*backward_gpu)  (struct layer, struct network);  
    void (*update_gpu)    (struct layer, update_args);  
    int batch_normalize;  // has batch normalization
    int shortcut;  
    int batch;  // number of batch
    int forced;  
    int flipped;  
    int inputs;  // number of elements in inputs: l.inputs = h*w*c;
    int outputs;  // number of elements in outputs: l.outputs = l.out_h * l.out_w * l.out_c;
    int nweights;  
    int nbiases;  
    int extra;  
    int truths;  
    int h,w,c;  // input h,w,c
    int out_h, out_w, out_c;  // output h,w,c
    int n;  // number of filters
    int max_boxes;  
    int groups;  
    int size;  // filter size
    int side;  
    int stride;  
    int reverse;  
    int flatten;  
    int spatial;  
    int pad;  
    int sqrt;  
    int flip;  
    int index;  
    int binary;  
    int xnor;  
    int steps;  
    int hidden;  
    int truth;  
    float smooth;  
    float dot;  
    float angle;  
    float jitter;  
    float saturation;  
    float exposure;  
    float shift;  
    float ratio;  
    float learning_rate_scale;  
    int softmax;  
    int classes;  
    int coords;  
    int background;  
    int rescore;  
    int objectness;  
    int does_cost;  
    int joint;  
    int noadjust;  
    int reorg;  
    int log;  
    int tanh;  

    float alpha;  
    float beta;  
    float kappa;  

    float coord_scale;  
    float object_scale;  
    float noobject_scale;  
    float mask_scale;  
    float class_scale;  
    int bias_match;  
    int random;  
    float thresh;  
    int classfix;  
    int absolute;  

    int onlyforward;  
    int stopbackward;  
    int dontload;  
    int dontloadscales;  

    float temperature;  
    float probability;  
    float scale;  

    char  * cweights;  
    int   * indexes;  
    int   * input_layers;  
    int   * input_sizes;  
    int   * map;  
    float * rand;  
    float * cost;  
    float * state;  
    float * prev_state;  
    float * forgot_state;  
    float * forgot_delta;  
    float * state_delta;  
    float * combine_cpu;  
    float * combine_delta_cpu;  

    float * concat;  
    float * concat_delta;  

    float * binary_weights;  

    float * biases;  // bias for conv if no BN, other wise it is gamma for BN
    float * bias_updates;  

    float * scales;  // beta for BN
    float * scale_updates;

    float * weights;  // conv weights
    float * weight_updates;

    float * delta; // in_grad
    float * output; // out_data
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;  // BN moving average
    float * rolling_variance;  // BN moving variance

    float * x;
    float * x_norm;

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;
// omit GPU and cudnn
};
```
