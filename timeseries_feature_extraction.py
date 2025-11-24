from tsfresh import extract_features 
import pandas as pd

def extract_timeseries_features(np_train, np_test):
    """
    Extract features from raw time series data using TSFRESH.

    Parameters:
    np_train (numpy.ndarray): Training data of shape (num_samples, num_timesteps, num_channels).
    np_test (numpy.ndarray): Testing data of shape (num_samples, num_timesteps, num_channels).

    Returns:
    X_train_features (pandas.DataFrame): Extracted features for training data.
    X_test_features (pandas.DataFrame): Extracted features for testing data.
    """
    def to_tsfresh_format(np_data):
        num_samples, num_timesteps, num_channels = np_data.shape
        df_list = []
        for sample_idx in range(num_samples):
            for channel_idx in range(num_channels):
                df_channel = pd.DataFrame({
                    'id': sample_idx,
                    'time': range(num_timesteps),
                    'value': np_data[sample_idx, :, channel_idx]
                })
                df_channel['channel'] = channel_idx
                df_list.append(df_channel)
        return pd.concat(df_list, ignore_index=True)

    df_train = to_tsfresh_format(np_train)
    df_test = to_tsfresh_format(np_test)

    X_train_features = extract_features(df_train, column_id='id', column_sort='time', column_kind='channel', column_value='value')
    X_test_features = extract_features(df_test, column_id='id', column_sort='time', column_kind='channel', column_value='value')

    return X_train_features, X_test_features