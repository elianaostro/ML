def handle_missing_values(df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame using a simplified k-Nearest Neighbors approach.

    For each column with missing values, it uses other numerical columns from the
    `reference_df` (or `df` itself if `reference_df` is None) to find the
    k-nearest neighbors of the rows with missing values in `df`, and imputes
    with the mean of the neighbors' values in that column from `reference_df`.

    Args:
        df (pd.DataFrame): DataFrame with potential missing values to be imputed.
        reference_df (Optional[pd.DataFrame], optional): DataFrame to use as a
            reference for finding nearest neighbors. If None, `df` is used.
            Defaults to None.
        n_neighbors (int, optional): The number of nearest neighbors to consider
            for imputation. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed using k-NN.
    """
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns

    if train_df is None:
        train_df = df_imputed

    ref_numeric_cols = train_df.select_dtypes(include=np.number).columns
    common_numeric_cols = list(set(numeric_cols) & set(ref_numeric_cols))

    for col_to_impute in numeric_cols:
        if df_imputed[col_to_impute].isnull().any():
            missing_mask = df_imputed[col_to_impute].isnull()
            missing_data = df_imputed[missing_mask][common_numeric_cols].values
            missing_indices = df_imputed[missing_mask].index

            observed_ref_data = train_df[train_df[col_to_impute].notna()][common_numeric_cols].values
            observed_ref_indices = train_df[train_df[col_to_impute].notna()].index

            if observed_ref_data.shape[0] >= n_neighbors and missing_data.shape[0] > 0:
                distances = cdist(missing_data, observed_ref_data, metric='euclidean')
                nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :n_neighbors]

                for i, missing_row_idx in enumerate(missing_indices):
                    neighbor_values = train_df.loc[observed_ref_indices[nearest_neighbor_indices[i]], col_to_impute].values
                    imputed_value = np.nanmean(neighbor_values)
                    df_imputed.loc[missing_row_idx, col_to_impute] = imputed_value

            elif missing_data.shape[0] > 0:
                fill_value = train_df[col_to_impute].mean()
                if pd.isna(fill_value):
                    fill_value = 0.0
                df_imputed.loc[missing_indices, col_to_impute] = fill_value

    return df_imputed


def remove_outliers(df: pd.DataFrame, underlimit: float = 0.15, uperlimit: float = 0.85) -> pd.DataFrame:
    """
    Identifies and removes outliers from within each cluster (with 2 clusters)
    for each numeric feature (excluding specified and binary-like) using a custom
    K-Means function. Outliers within each cluster are identified using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        underlimit (float): Lower percentile for IQR calculation. Defaults to 0.15.
        uperlimit (float): Upper percentile for IQR calculation. Defaults to 0.85.

    Returns:
        pd.DataFrame: DataFrame with outliers within each cluster set to NaN.
    """
    df_no_outliers = df.copy()
    numeric_cols = df_no_outliers.select_dtypes(include=np.number).columns.difference(['CellAdhesion', 'NuclearMembrane', 'OxygenSaturation', 'Vascularization', 'InflammationMarkers'])
    numeric_cols = [col for col in numeric_cols if not set(df_no_outliers[col].dropna().unique()).issubset({0, 1})]

    iqr_multiplier = 1.5  

    for column in numeric_cols:
        data_col_df = df_no_outliers[[column]].dropna()
        if len(data_col_df) < 2: 
            continue

        try:
            labels, _ = KMeans(data_col_df, n_clusters=2, random_state=42)
            clusters = pd.Series(labels, index=data_col_df.index)
        except ValueError as e:
            print(f"Error during custom KMeans for column {column}: {e}")
            continue

        for cluster_label in np.unique(clusters):
            cluster_data = data_col_df[clusters == cluster_label]

            if len(cluster_data) > 2:  
                Q1 = cluster_data[column].quantile(underlimit)
                Q3 = cluster_data[column].quantile(uperlimit)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR

                outlier_indices_cluster = cluster_data.index[(cluster_data[column] < lower_bound) | (cluster_data[column] > upper_bound)]

                df_no_outliers.loc[outlier_indices_cluster, column] = np.nan

    return df_no_outliers