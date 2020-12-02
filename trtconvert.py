#encoding=utf-8
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import sys, os, argparse, logging, random, cv2, glob
# logging.basicConfig(level=logging.NOTSET)
import common
import graphsurgeon as gs
import uff
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()

parser.add_argument('--fp16', action="store_true")
parser.add_argument('--int8', action="store_true")

parser.add_argument('--onnx', action="store_true")
parser.add_argument('--caffe', action="store_true")
parser.add_argument('--pb', action="store_true")
parser.add_argument('--nchw')
parser.add_argument('--dynamic', action="store_true")
parser.add_argument('--cache', action="store_true")

args = parser.parse_args()

if not args.cache:
    os.system("rm calibration.cache")

[batch_size, channel, height, width] = list(map(int,args.nchw.split(",")))
print("batchsize:", batch_size, " channel:", channel, " height:", height, " width:", width)

NUM_BATCHES = 0
NUM_PER_BATCH = 1
NUM_CALIBRATION_IMAGES = 1000

DIR_NAME = "./input_model"
LIB_FILE = os.path.abspath(os.path.join(DIR_NAME, 'libflattenconcat.so'))
MODEL_SPECS = {
    'ssd_mobilenet_v1_coco': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_coco.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_coco.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v1_coco.bin')),
        'num_classes': 91,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
    'ssd_mobilenet_v1_egohands': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_egohands.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v1_egohands.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v1_egohands.bin')),
        'num_classes': 2,
        'min_size': 0.05,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
    'ssd_mobilenet_v2_coco': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v2_coco.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v2_coco.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v2_coco.bin')),
        'num_classes': 91,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [1, 0, 2],  # order of loc_data, conf_data, priorbox_data
    },
    'ssd_mobilenet_v2_egohands': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v2_egohands.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_mobilenet_v2_egohands.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_mobilenet_v2_egohands.bin')),
        'num_classes': 2,
        'min_size': 0.05,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
    'ssd_inception_v2_coco': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_inception_v2_coco.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'ssd_inception_v2_coco.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssd_inception_v2_coco.bin')),
        'num_classes': 91,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
    'ssdlite_mobilenet_v2_coco': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'ssdlite_mobilenet_v2_coco.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'ssdlite_mobilenet_v2_coco.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'TRT_ssdlite_mobilenet_v2_coco.bin')),
        'num_classes': 91,
        'min_size': 0.2,
        'max_size': 0.95,
        'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
    },
}

def prepare_namespace_plugin_map():
    # plugin 替换
    resize = gs.create_plugin_node(name="trt_resize", op="ResizeNearest_TRT", scale=2.0)
    namespace_plugin_map = {
        "resize_1/ResizeBilinear": resize
    }
    return namespace_plugin_map

def replace_addv2(graph):
    """Replace all 'AddV2' in the graph with 'Add'.

    'AddV2' is not supported by UFF parser.

    Reference:
    1. https://github.com/jkjung-avt/tensorrt_demos/issues/113#issuecomment-629900809
    """
    for node in graph.find_nodes_by_op('AddV2'):
        gs.update_node(node, op='Add')
    return graph

def replace_fusedbnv3(graph):
    """Replace all 'FusedBatchNormV3' in the graph with 'FusedBatchNorm'.

    'FusedBatchNormV3' is not supported by UFF parser.

    Reference:
    1. https://devtalk.nvidia.com/default/topic/1066445/tensorrt/tensorrt-6-0-1-tensorflow-1-14-no-conversion-function-registered-for-layer-fusedbatchnormv3-yet/post/5403567/#5403567
    2. https://github.com/jkjung-avt/tensorrt_demos/issues/76#issuecomment-607879831
    """
    for node in graph.find_nodes_by_op('FusedBatchNormV3'):
        gs.update_node(node, op='FusedBatchNorm')
    return graph

def add_anchor_input(graph):
    """Add the missing const input for the GridAnchor node.

    Reference:
    1. https://www.minds.ai/post/deploying-ssd-mobilenet-v2-on-the-nvidia-jetson-and-nano-platforms
    """
    data = np.array([1, 1], dtype=np.float32)
    anchor_input = gs.create_node('AnchorInput', 'Const', value=data)
    graph.append(anchor_input)
    graph.find_nodes_by_op('GridAnchor_TRT')[0].input.insert(0, 'AnchorInput')
    return graph

def add_plugin(graph, model, spec):
    """add_plugin

    Reference:
    1. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
    2. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v2_coco_2018_03_29.py
    3. https://devtalk.nvidia.com/default/topic/1050465/jetson-nano/how-to-write-config-py-for-converting-ssd-mobilenetv2-to-uff-format/post/5333033/#5333033
    """
    numClasses = spec['num_classes']
    minSize = spec['min_size']
    maxSize = spec['max_size']
    inputOrder = spec['input_order']

    all_assert_nodes = graph.find_nodes_by_op('Assert')
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op('Identity')
    graph.forward_inputs(all_identity_nodes)
    INPUT_DIMS = (3, 300, 300)
    Input = gs.create_plugin_node(
        name='Input',
        op='Placeholder',
        shape=(1,) + INPUT_DIMS
    )

    PriorBox = gs.create_plugin_node(
        name='MultipleGridAnchorGenerator',
        op='GridAnchor_TRT',
        minSize=minSize,  # was 0.2
        maxSize=maxSize,  # was 0.95
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1, 0.1, 0.2, 0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name='NMS',
        op='NMS_TRT',
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,  # was 1e-8
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=numClasses,  # was 91
        inputOrder=inputOrder,
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        'concat_priorbox',
        op='ConcatV2',
        axis=2
    )

    if trt.__version__[0] >= '7':
        concat_box_loc = gs.create_plugin_node(
            'concat_box_loc',
            op='FlattenConcat_TRT',
            axis=1,
            ignoreBatch=0
        )
        concat_box_conf = gs.create_plugin_node(
            'concat_box_conf',
            op='FlattenConcat_TRT',
            axis=1,
            ignoreBatch=0
        )
    else:
        concat_box_loc = gs.create_plugin_node(
            'concat_box_loc',
            op='FlattenConcat_TRT'
        )
        concat_box_conf = gs.create_plugin_node(
            'concat_box_conf',
            op='FlattenConcat_TRT'
        )

    namespace_for_removal = [
        'ToFloat',
        'image_tensor',
        'Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3',
    ]
    namespace_plugin_map = {
        'MultipleGridAnchorGenerator': PriorBox,
        'Postprocessor': NMS,
        'Preprocessor': Input,
        'ToFloat': Input,
        'Cast': Input,  # added for models trained with tf 1.15+
        'image_tensor': Input,
        'MultipleGridAnchorGenerator/Concatenate': concat_priorbox,  # for 'ssd_mobilenet_v1_coco'
        'Concatenate': concat_priorbox,  # for other models
        'concat': concat_box_loc,
        'concat_1': concat_box_conf
    }

    graph.remove(graph.find_nodes_by_path(['Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3']), remove_exclusive_dependencies=False)  # for 'ssd_inception_v2_coco'

    graph.collapse_namespaces(namespace_plugin_map)
    graph = replace_addv2(graph)
    graph = replace_fusedbnv3(graph)

    if 'image_tensor:0' in graph.find_nodes_by_name('Input')[0].input:
        graph.find_nodes_by_name('Input')[0].input.remove('image_tensor:0')
    if 'Input' in graph.find_nodes_by_name('NMS')[0].input:
        graph.find_nodes_by_name('NMS')[0].input.remove('Input')
    # Remove the Squeeze to avoid "Assertion 'isPlugin(layerName)' failed"
    graph.forward_inputs(graph.find_node_inputs_by_name(graph.graph_outputs[0], 'Squeeze'))
    if 'anchors' in [node.name for node in graph.graph_outputs]:
        graph.remove('anchors', remove_exclusive_dependencies=False)
    if len(graph.find_nodes_by_op('GridAnchor_TRT')[0].input) < 1:
        graph = add_anchor_input(graph)
    if 'NMS' not in [node.name for node in graph.graph_outputs]:
        graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
        # if 'NMS' not in [node.name for node in graph.graph_outputs]:
        #     # We expect 'NMS' to be one of the outputs
        #     raise RuntimeError('bad graph_outputs')

    return graph

def preprocess(image):
    """ 预处理代码。

    设置均值、方差、转置等操作。
    """
    data = cv2.imread(image)
    data = cv2.resize(data, (224, 224))/ 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    data = (data - mean) / std
    data = np.transpose(data, (2, 0, 1))
    return data 

def load_data(data_dir):
    """ load_data 读取数据。
        不同的模型需要根据其数据预处理进行修改。      
    """
    imgs = glob.glob(data_dir+'/*.jpg')
    NUM_BATCHES = NUM_CALIBRATION_IMAGES // NUM_PER_BATCH + (NUM_CALIBRATION_IMAGES % NUM_PER_BATCH > 0)
    print("NUM_BATCHES: ", NUM_BATCHES)
    img = 0
    batches = np.zeros(shape=(NUM_BATCHES,channel,height,width),dtype=np.float32)
    for i in range(NUM_BATCHES-1):
        batch = np.zeros(shape=(NUM_PER_BATCH,channel,height,width),dtype=np.float32)
        for j in range(NUM_PER_BATCH):
            image_data = preprocess(imgs[img])
            batch[j] += image_data
            img += 1
        batches[i] = batch
    return batches


class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Int8EntropyCalibrator 类
    """
    def __init__(self, cache_file, batch_size):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = load_data("./calibration_data")
        self.batch_size = batch_size
        self.current_index = 0
        # 分配足够的内存给整个 batch
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    # 返回 batch size 大小。
    def get_batch_size(self):
        return self.batch_size

    # TensorTR 将 engine 的 bindings 的 names 传给 get_batch 函数。
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    # 读取
    def read_calibration_cache(self):
        # 如果校准 cache 文件存在，则使用 cache 文件而不是再次校准。否则，直接返回 None。
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # 将校准得到的数据写入cache文件。
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine():
    TRT_LOGGER = trt.Logger()
    builder = trt.Builder(TRT_LOGGER)

    # ONNX 模型转换
    if args.onnx:
        # EXPLICIT_BATCH 用于解析 Onnx 模型
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network=network, logger=TRT_LOGGER)

        model_path = os.listdir("./input_model")[0]
        model_file = os.path.join('./input_model',model_path)
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        # print("Number of layers: " + network.num_layers)
        last_layer = network.get_layer(network.num_layers - 1)
        # print(last_layer.get_output(0))
        if not last_layer.get_output(0):
          network.mark_output(last_layer.get_output(0))
        
    # Caffe 模型转换
    elif args.caffe:
        network = builder.create_network()
        parser = trt.CaffeParser()
        model_path = os.listdir("./input_model")
        model_file = os.path.join('./input_model',model_path[0])
        [caffemodel] = [i for i in model_path if i.split('.')[-1] == "caffemodel"]
        caffemodel = os.path.join("./input_model",caffemodel)
        [prototxt] = [i for i in model_path if i.split('.')[-1] == "prototxt"]
        prototxt = os.path.join('./input_model', prototxt)
        logging.info("Caffe model files:"+prototxt+caffemodel)
        model_tensors = parser.parse(deploy=prototxt, model=caffemodel, network=network, dtype=trt.float32)
        if model_tensors:
            logging.info("Model file parse success!")
        else:
            logging.info("Model file parse failed!")
            return None
        last_layer = network.get_layer(network.num_layers - 1)
        network.mark_output(last_layer.get_output(0))

    # PB 模型转换
    elif args.pb:
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        network = builder.create_network()
        parser = trt.UffParser()

        model_path = os.listdir("./input_model")[0]
        model_file = os.path.join('./input_model',model_path)

        spec = MODEL_SPECS[model_path.split(".")[0]]
        dynamic_graph = add_plugin(
            gs.DynamicGraph(model_file),
            model_path.split(".")[0],
            spec)
        _ = uff.from_tensorflow(
            dynamic_graph.as_graph_def(),
            output_nodes=['NMS'],
            output_filename=spec['tmp_uff'],
            text=True,
            debug_mode=False)
        parser.register_input("Input",(channel,height,width))
        parser.register_output("MarkOutput_0")
        parser.parse(spec['tmp_uff'], network)
        os.system("rm ./input_model/*.pbtxt")
        os.system("rm ./input_model/*.uff")
        
    # 不支持模型类型
    else:
        logging.error("Unsupported Model Type!")
        return None


    config = builder.create_builder_config()
    config.max_workspace_size = common.GiB(5)
    if args.dynamic:
        print("Dynamic model setting optimization profile")
        profile = builder.create_optimization_profile()
        profile.set_shape("data", (batch_size,channel,height,width), (batch_size,channel,height,width), (batch_size,channel,height,width)) 
        config.add_optimization_profile(profile)

    builder.max_batch_size = batch_size
    # builder.max_workspace_size = common.GiB(5)
    if args.int8:
        if builder.platform_has_fast_int8:
            print("Convert in INT8 mode.")
            print("Platform has fast int8.")
        # builder.int8_mode = 1
        # builder.int8_calibrator = Int8EntropyCalibrator(cache_file="calibration.cache", batch_size=batch_size)
        
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8EntropyCalibrator(cache_file="calibration.cache", batch_size=batch_size)

    elif args.fp16:
        if builder.platform_has_fast_fp16:
            print("Convert in FP16 mode.")
            print("Platform has fast fp16.")
        config.set_flag(trt.BuilderFlag.FP16)

    else:
        if builder.platform_has_tf32:
            print("Convert in TF32 mode.")
            print("Platform has tf32.")

    engine = builder.build_engine(network, config)
    # engine = builder.build_cuda_engine(network)
    if engine:
        if args.int8:
            trt_engine_name = model_file.split("/")[-1].split(".")[0]+"_int8.trt"
        elif args.fp16:
            trt_engine_name = model_file.split("/")[-1].split(".")[0]+"_fp16.trt"
        else:
            trt_engine_name = model_file.split("/")[-1].split(".")[0]+"_fp32.trt"
        print('Saving TRT engine file to path {}...'.format(trt_engine_name))
        with open(os.path.join('./output_trt_models',trt_engine_name), "wb") as f:
            f.write(engine.serialize())
            print(bytes(engine.serialize()).__sizeof__)
            # f.write(sys.getsizeof(engine.serialize()))
        print('Engine file has already saved to {}!'.format(trt_engine_name))
    else:
        logging.error("Model engine build fail.")

    return engine

def main():
    engine = build_engine()
    if not engine:
        return False
    context = engine.create_execution_context()
    # context.active_optimization_profile = 0
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    image_data = preprocess("./test_data/allman2.jpg")
    # image_data = image_data[np.newaxis, :, :, :].astype(np.float32)
    image_data = image_data.ravel().astype(np.float32)
    print("Test image shape:", image_data.shape)
    inputs[0].host = image_data
    if args.caffe or args.pb:
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    else:
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(trt_outputs)

if __name__ == "__main__":
    main()