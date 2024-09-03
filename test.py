import numpy as np
from keras.models import model_from_json
import numpy as np
with open("model.json") as file:
    model = file.read()
model = model_from_json(model)
model.load_weights("model.h5")

inp =np.array([[0,0],[0,1],[1,0],[1,1]])
res = model.predict(inp)
print(np.round(res))
