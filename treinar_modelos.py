import pickle

with open("modelo_libras_expandido.pkl", "rb") as f:
    model = pickle.load(f)

with open("modelo_info_expandido.pkl", "rb") as f:
    info = pickle.load(f)

print("✅ Modelo carregado")
print("Classes:", info["classes"])
print("Features esperadas:", info["n_features"])
