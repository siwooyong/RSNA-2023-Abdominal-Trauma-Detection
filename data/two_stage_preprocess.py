from sklearn.model_selection import KFold

def preprocess():
    train_series_meta = pd.read_csv('/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/DataSources/train_series_meta.csv')
    train_series_meta = train_series_meta.sort_values(by='patient_id').reset_index(drop=True)
    train = pd.read_csv('/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/DataSources/train.csv')
    train = train.sort_values(by='patient_id').reset_index(drop=True)
    _train = []
    series_ids = []
    for i in range(len(train_series_meta)):
      patient_id, series_id, _, _ = train_series_meta.loc[i]
      sample = train[train['patient_id']==patient_id]
      _train.append(sample)
      series_ids.append(int(series_id))

    _train = pd.concat(_train).reset_index(drop=True)
    _train['series_id'] = series_ids

    train = _train

    injury_train = train[train['any_injury']==1].reset_index(drop=True)
    normal_train = train[train['any_injury']==0].reset_index(drop=True)

    kf = KFold(n_splits=5)
    injury_folds = []
    for i, (train_index, test_index) in enumerate(kf.split(injury_train)):
      train_df = injury_train.loc[train_index]
      val_df = injury_train.loc[test_index]
      injury_folds.append([train_df, val_df])


    kf = KFold(n_splits=5)
    normal_folds = []
    for i, (train_index, test_index) in enumerate(kf.split(normal_train)):
      train_df = normal_train.loc[train_index]
      val_df = normal_train.loc[test_index]
      normal_folds.append([train_df, val_df])
    return train, injury_folds, normal_folds
