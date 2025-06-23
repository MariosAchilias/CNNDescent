import cnndescent
import struct

index = cnndescent.CNNIndex(100, 100, cnndescent.Dist.EUCLIDEAN)

with open("c_lib/00050000-1.bin", 'rb') as file:
	rawDataset = file.read()[4:] # reads an extra 4 byte number at beginning of file (why?)

dim = 100
stride = struct.calcsize("<" + "f" * dim)
for i in range(0, round(len(rawDataset) / struct.calcsize("<" + "f" * dim))):
	start = i * stride
	point = struct.unpack("<" + "f" * 100, rawDataset[start:start + stride])
	index.add_point(list(point))

index.build_index_nndescent(0.001, 0.5, 20)