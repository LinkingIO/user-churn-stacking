
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import os

features_all = ["user_id",'cid',"device_type","fix_month","gender","current_role","last_lat","last_lon","hometown","profession","source_channel","work_city_id","home_city_id","work_lon","home_lat","work_lat","home_lon","age_section","age","version","model", 'label']
dtype_all = {
    'user_id': 'int64',
    'device_type': 'int8',
    'fix_month': 'int8',
    'gender': 'int8',
    'current_role': 'int8',
    'last_lat': 'float32',
    'last_lon': 'float32',
    'hometown': 'category',
    'profession': 'category',
    'source_channel': 'category',
    'work_city_id': 'int8',
    'home_city_id': 'int8',
    'work_lon': 'float32',
    'home_lat': 'float32',
    'work_lat': 'float32',
    'home_lon': 'float32',
    'age_section': 'category',
    'age': 'int8',
    'version': 'category',
    'model': 'category',
    'label': 'int8'
}

features = ["user_id","device_type","fix_month","gender","current_role","last_lat","last_lon","hometown","profession","source_channel","work_city_id","home_city_id","work_lon","home_lat","work_lat","home_lon","age_section","age","version","model"]
# set the column data types
dtype = {
    'user_id': 'int64',
    'device_type': 'int8',
    'fix_month': 'int8',
    'gender': 'int8',
    'current_role': 'int8',
    'last_lat': 'float32',
    'last_lon': 'float32',
    'hometown': 'category',
    'profession': 'category',
    'source_channel': 'category',
    'work_city_id': 'int8',
    'home_city_id': 'int8',
    'work_lon': 'float32',
    'home_lat': 'float32',
    'work_lat': 'float32',
    'home_lon': 'float32',
    'age_section': 'category',
    'age': 'int8',
    'version': 'category',
    'model': 'category'
}


def check_path(_path):
    """Check weather the _path exists. If not, make the dir."""
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
            os.makedirs(os.path.dirname(_path))


def get_input_df(path):
    # create a DataFrame from the dictionary
    data = pd.read_csv(path, delimiter="\t", names=features_all).drop("cid", axis=1)

    # apply the column data types to the DataFrame
    df = data.astype(dtype_all)

    # fix the issue of NA field
    df['hometown'] = df['hometown'].cat.add_categories('NA')
    df['hometown'] = df['hometown'].fillna('NA')
    df['profession'] = df['profession'].cat.add_categories('NA')
    df['profession'] = df['profession'].fillna('NA')

    return df


def feature_analyze(model, to_print=False, to_plot=False, csv_path=None):
    """XGBOOST 模型特征重要性分析。

    Args:
        model: 训练好的 xgb 模型。
        to_print: bool, 是否输出每个特征重要性。
        to_plot: bool, 是否绘制特征重要性图表。
        csv_path: str, 保存分析结果到 csv 文件路径。
    """
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    if to_plot:
        features = list()
        scores = list()
        for (key, value) in feature_score:
            features.append(key)
            scores.append(value)
        plt.barh(range(len(scores)), scores)
        plt.yticks(range(len(scores)), features)
        for i in range(len(scores)):
            plt.text(scores[i] + 0.75, i - 0.25, scores[i])
        plt.xlabel('feature socre')
        plt.title('feature score evaluate')
        plt.grid()
        plt.show()
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    if to_print:
        print(''.join(fs))
    if csv_path is not None:
        with open(csv_path, 'w') as f:
            f.writelines("feature,score\n")
            f.writelines(fs)
    return feature_score