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
parser.add_argument('--batchsize')
parser.add_argument('--channel')
parser.add_argument('--height')
parser.add_argument('--width')
parser.add_argument('--dynamic', action="store_true")

args = parser.parse_args()

batch_size = int(args.batchsize)
height = int(args.height)
width = int(args.width)
channel = int(args.channel)

NUM_BATCHES = 0
NUM_PER_BATCH = 1
NUM_CALIBRATION_IMAGES = 800

def preprocess(image):
    data = cv2.imread(image)
    data = cv2.resize(data, (224, 224))/ 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    data = (data - mean) / std
    data = np.transpose(data, (2, 0, 1))
    return data 

# load_data 读取数据。
# 不同的模型需要根据其数据预处理进行修改。
def load_data(data_dir):
    
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

# Int8EntropyCalibrator 类
class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
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

def prepare_namespace_plugin_map():
    # plugin 替换
    resize = gs.create_plugin_node(name="trt_resize", op="ResizeNearest_TRT", scale=2.0)
    namespace_plugin_map = {
        "resize_1/ResizeBilinear": resize
    }
    return namespace_plugin_map

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
        network = builder.create_network()
        parser = trt.UffParser()

        model_path = os.listdir("./input_model")[0]
        model_file = os.path.join('./input_model',model_path)

        dynamic_graph = gs.DynamicGraph(model_file)
        dynamic_graph.collapse_namespaces(prepare_namespace_plugin_map())
        # Save resulting graph to UFF file
        output_uff_path = os.path.splitext(model_file)[0] + ".uff"
        uff.from_tensorflow(
            dynamic_graph.as_graph_def(),
            ["scores"],
            output_filename=os.path.join('./input_model',output_uff_path),
            text=True
        )
        parser.register_input("tf_example",(batch_size,channel,height,width))
        parser.register_output("scores")
        model_file = os.path.join('./input_model',output_uff_path)
        parser.parse(model_file, network)

        last_layer = network.get_layer(network.num_layers - 1)
        if not last_layer.get_output(0):
            network.mark_output(last_layer.get_output(0))
        
    # 不支持模型类型
    else:
        logging.error("Unsupported Model Type!")
        return None


    config = builder.create_builder_config()
    if args.dynamic:
        print("Dynamic model setting optimization profile")
        profile = builder.create_optimization_profile()
        profile.set_shape("data", (batch_size,channel,height,width), (batch_size,channel,height,width), (batch_size,channel,height,width)) 
        config.add_optimization_profile(profile)

    builder.max_batch_size = batch_size
    builder.max_workspace_size = common.GiB(5)
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
    if args.caffe:
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    else:
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(trt_outputs)

if __name__ == "__main__":
    main()

    
