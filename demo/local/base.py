
from src import generate_colors
from src import read_json
from pathlib import Path
from src import MixinBlockPolymer, Cells

data_folder = Path.cwd().parent / "datasets"

input_dict = read_json(data_folder / "scatter" / "input.json")

mx = MixinBlockPolymer(input_dict)
print(mx["fA"])
# print([mx[i] for i in ["lx", "ly" ,"lz"]])
mx["lx"] = 8
print(mx.get("lx", "ly" ,"lz"))



# print(generate_colors(mode="HEX", num=1))

# print(Reader.read_printout(data_folder / "scatter"))

# print(Reader.read_phout(data_folder / "scatter"))

# res = Reader.read_density(data_folder / "Vori")
# print(res.reshaped[1].shape)
#
# res.shape = res.shape[::-1]
# print(res.reshaped[1].shape)