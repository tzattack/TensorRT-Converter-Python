export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

apt-get update
apt-get install python3-pip

pip3 install tensorrt-7.2.0.14-cp36-none-linux_x86_64.whl
pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl
pip3 install uff-0.6.5-py2.py3-none-any.whl
sudo pip3 install requirements.txt

# install pycuda if necessary
if ! python3 -c "import pycuda" > /dev/null 2>&1; then
  ./install_pycuda.sh
fi

echo "** Patch 'graphsurgeon.py' in TensorRT"

script_path=$(realpath $0)
#gs_path=$(ls /usr/lib/python3.?/dist-packages/graphsurgeon/node_manipulation.py)
gs_path=$(ls /usr/local/lib/python3.6/dist-packages/graphsurgeon/node_manipulation.py)
patch_path=$(dirname $script_path)/graphsurgeon.patch

if head -30 ${gs_path} | tail -1 | grep -q NodeDef; then
  # This is for JetPack-4.2
  patch -N -p1 -r - ${gs_path} ${patch_path}-4.2 && echo
fi
if head -22 ${gs_path} | tail -1 | grep -q update_node; then
  # This is for JetPack-4.2.2
  patch -N -p1 -r - ${gs_path} ${patch_path}-4.2.2 && echo
fi
if head -69 ${gs_path} | tail -1 | grep -q update_node; then
  # This is for JetPack-4.4
  patch -N -p1 -r - ${gs_path} ${patch_path}-4.4 && echo
fi

echo "** Making symbolic link of libflattenconcat.so"

trt_version=$(echo /usr/lib/aarch64-linux-gnu/libnvinfer.so.? | cut -d '.' -f 3)
if [ "${trt_version}" = "5" ] || [ "${trt_version}" = "6" ]; then
  ln -sf libflattenconcat.so.${trt_version} libflattenconcat.so
fi

echo "** Installation done"
