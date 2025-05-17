import os
import pandas as pd
input_dir = "E:/datasets/SOTA/test/audio"

file_list = []
label_list = []
for file in os.listdir(input_dir):
    if file.endswith(".wav"):
        file_list.append(file)
        tts = file.split(".")[0]
        # if splittable by underscore, then mark it as spoof, or else mark it as bonafide
        if "_" in tts:
            label = "spoof"
        else:
            label = "bonafide"
        label_list.append(tts)

df = pd.DataFrame(file_list, columns=["file_name", "label"])

for file, label in zip(file_list, label_list):
    # add file_name and label to the dataframe
    df = df.append({"file_name": file, "label": label}, ignore_index=True)

# save the dataframe to a csv file
df.to_csv("E:/datasets/SOTA/test/meta.csv", index=False)