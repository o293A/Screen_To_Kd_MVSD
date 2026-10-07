"""Converts EasyOCR's english_g2.pth into assets/crnn_weights.bin (flat little-endian f32).

BatchNorm layers are fused into the preceding convolutions. Needs only numpy
(no PyTorch): the .pth zip is read with a tiny custom unpickler.
Usage: python3 convert_weights.py english_g2.pth ../assets/crnn_weights.bin
"""
import sys, zipfile, pickle, io, collections
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
prefix = pk[: -len("data.pkl")]
arrays = {}

class Storage:  # placeholder object returned by persistent_load
    pass

class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_rebuild_tensor_v2":
            def rebuild(storage, off, size, stride, *a):
                flat = arrays[id(storage)]
                n = int(np.prod(size)) if size else 1
                base = flat[off:]
                return np.lib.stride_tricks.as_strided(
                    base, shape=size, strides=[s * base.itemsize for s in stride]
                ).copy() if size else base[:1].copy()
            return rebuild
        if name == "OrderedDict":
            return collections.OrderedDict
        if name.endswith("Storage"):
            return name
        if name == "_rebuild_parameter":
            return lambda data, rg, bh, *a: data
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        _, st, key, _, _ = pid
        st = st if isinstance(st, str) else st.__name__
        dt = {"FloatStorage": np.float32, "LongStorage": np.int64}[st]
        o = Storage()
        arrays[id(o)] = np.frombuffer(z.read(prefix + "data/" + str(key)), dtype=dt)
        return o

sd = U(io.BytesIO(z.read(pk))).load()
sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

def fuse(conv, bn):
    w = sd[conv + ".weight"]
    scale = sd[bn + ".weight"] / np.sqrt(sd[bn + ".running_var"] + 1e-5)
    w2 = w * scale[:, None, None, None]
    b2 = sd[bn + ".bias"] - sd[bn + ".running_mean"] * scale
    return w2, b2

out = []
def put(a): out.append(np.ascontiguousarray(a, dtype="<f4").ravel())

for i in (0, 3, 6, 8):
    put(sd[f"FeatureExtraction.ConvNet.{i}.weight"]); put(sd[f"FeatureExtraction.ConvNet.{i}.bias"])
for c, b in ((11, 12), (14, 15)):
    w, bias = fuse(f"FeatureExtraction.ConvNet.{c}", f"FeatureExtraction.ConvNet.{b}")
    put(w); put(bias)
put(sd["FeatureExtraction.ConvNet.18.weight"]); put(sd["FeatureExtraction.ConvNet.18.bias"])
for l in (0, 1):
    p = f"SequenceModeling.{l}."
    for suffix in ("", "_reverse"):
        put(sd[p + "rnn.weight_ih_l0" + suffix])
        put(sd[p + "rnn.weight_hh_l0" + suffix])
        put(sd[p + "rnn.bias_ih_l0" + suffix] + sd[p + "rnn.bias_hh_l0" + suffix])
    put(sd[p + "linear.weight"]); put(sd[p + "linear.bias"])
put(sd["Prediction.weight"]); put(sd["Prediction.bias"])
blob = np.concatenate(out)
blob.tofile(dst)
print("floats:", blob.size, "bytes:", blob.nbytes)
