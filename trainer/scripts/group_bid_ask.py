import os
import re


for data_type in os.listdir("./data"):
    print(f"type : {data_type}")
    group_map = dict()
    file_dir = "./data/" + data_type
    files = os.listdir(file_dir)
    for file in files:
        file_path = file_dir + "/" + file
        if os.path.isfile(file_path):
            
            match = re.search("_BID_", file)
            if match:
                start, end = match.span()
            else:
                match = re.search("_ASK_", file)
                if match:
                    start, end = match.span()
                else:
                    continue
            
            match = re.search(".csv", file)
            if match:
                ext, _ = match.span()
                data_group = file[:start] + "_" + file[end:ext]
                data_group_path = file_dir + "/" + data_group
                if group_map.get(data_group) is None:
                    print("creating dir: " + data_group_path)
                    os.makedirs(data_group_path)
                    os.rename(file_path, data_group_path + "/" + file)
                    group_map[data_group] = 1
                elif group_map.get(data_group) == 1:
                    os.rename(file_path, data_group_path + "/" + file)
            #os.makedirs("./data/" + data_type + "/" + file[:start] + file[end:ext])

    