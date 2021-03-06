from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import seaborn as sns

iris_dataset = load_iris()

df = pd.DataFrame.from_dict(np.hstack((iris_dataset["data"], iris_dataset["target"].reshape(-1, 1))))
df.columns = iris_dataset["feature_names"] + ["target"]
df["target_names"] = df["target"].apply(lambda x: iris_dataset["target_names"][int(x)])

# scatter plot
df.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)")

print("Plotting has been created")
