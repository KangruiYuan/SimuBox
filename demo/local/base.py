
from src import generate_colors
from src import Reader
from pathlib import Path


data_folder = Path.cwd().parent / "datasets"

# print(generate_colors(mode="HEX", num=1))

print(Reader.read_printout(data_folder / "scatter"))

# print(Reader.read_phout(data_folder / "scatter"))

res = Reader.read_density(data_folder / "Vori")
print(res.reshaped[1].shape)

res.shape = res.shape[::-1]
print(res.reshaped[1].shape)