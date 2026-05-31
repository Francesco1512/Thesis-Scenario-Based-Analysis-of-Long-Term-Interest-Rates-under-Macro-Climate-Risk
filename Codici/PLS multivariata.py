import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def run_pls(
    file_path,
    sheet_name,
    date_col,
    x_cols,
    y_cols,
    n_components=None,
    output_path=None,
    scale_y=True,
):
    """Run PLS on predictors in `x_cols` and responses in `y_cols`.

    Saves components, weights, coefficients and summary to Excel if `output_path` provided.
    Returns a dictionary with key results.
    """

    # Load data
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Isolate relevant columns
    use_cols = [date_col] + x_cols + y_cols
    df = df[use_cols].copy()
    
    # CRITICAL FIX: Coerce data to numeric to bypass the 'units' row (row 2 in your file)
    # This turns strings like "Percent per annum" into NaN, which are then cleanly dropped.
    df[x_cols + y_cols] = df[x_cols + y_cols].apply(pd.to_numeric, errors='coerce')
    
    # Drop NaNs and sort by date
    df = df.dropna().sort_values(date_col).reset_index(drop=True)

    dates = df[date_col].reset_index(drop=True)
    X = df[x_cols].values
    Y = df[y_cols].values

    # Determine components
    p = X.shape[1]
    q = Y.shape[1]
    if n_components is None:
        n_components = min(p, q, 3)

    # Standardize X and Y
    sx = StandardScaler()
    Xs = sx.fit_transform(X)

    if scale_y:
        sy = StandardScaler()
        Ys = sy.fit_transform(Y)
    else:
        Ys = Y.copy()

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(Xs, Ys)

    # Scores (components)
    X_scores = pls.x_scores_  # shape (n_samples, n_components)
    Y_scores = pls.y_scores_

    # Weights and loadings
    X_weights = pls.x_weights_  # weights used to build components
    X_loadings = pls.x_loadings_
    Y_loadings = pls.y_loadings_

    # Coefficients to predict Y from X (on scaled space)
    coef = pls.coef_

    # Predictions and R2 (on original Y scale)
    Ys_pred = pls.predict(Xs)
    if scale_y:
        Y_pred = sy.inverse_transform(Ys_pred)
    else:
        Y_pred = Ys_pred

    r2_per_response = {}
    for i, col in enumerate(y_cols):
        r2_per_response[col] = r2_score(Y[:, i], Y_pred[:, i])

    # Explained variance of X by components (fraction of X total variance)
    var_scores = np.var(X_scores, axis=0, ddof=1)
    total_var_X = np.sum(np.var(Xs, axis=0, ddof=1))
    explained_var = var_scores / total_var_X

    results = {
        "model": pls,
        "X_scaler": sx,
        "Y_scaler": sy if scale_y else None,
        "X_scores": X_scores,
        "Y_scores": Y_scores,
        "X_weights": X_weights,
        "X_loadings": X_loadings,
        "Y_loadings": Y_loadings,
        "coef": coef,
        "r2": r2_per_response,
        "explained_var_X": explained_var,
        "dates": dates,
        "x_cols": x_cols,
        "y_cols": y_cols,
    }

    if output_path is not None:
        # Build dataframes to save
        comp_df = pd.DataFrame(X_scores, columns=[f"Comp_{i+1}" for i in range(X_scores.shape[1])])
        comp_df.insert(0, date_col, dates)

        weights_df = pd.DataFrame(
            X_weights, index=x_cols, columns=[f"w_comp_{i+1}" for i in range(X_weights.shape[1])]
        )

        loadings_df = pd.DataFrame(
            X_loadings, index=x_cols, columns=[f"loading_comp_{i+1}" for i in range(X_loadings.shape[1])]
        )

        # coef may have shape (n_features, n_targets) or its transpose depending on sklearn version
        coef_arr = coef
        if coef_arr.shape == (len(y_cols), len(x_cols)):
            coef_arr = coef_arr.T
        if coef_arr.shape != (len(x_cols), len(y_cols)):
            raise ValueError(f"Unexpected coef shape {coef.shape}; expected ({len(x_cols)},{len(y_cols)})")
        coef_df = pd.DataFrame(coef_arr, index=x_cols, columns=y_cols)

        r2_df = pd.DataFrame.from_dict(r2_per_response, orient="index", columns=["R2"]) 

        explained_df = pd.DataFrame({
            "component": [f"Comp_{i+1}" for i in range(len(explained_var))],
            "explained_var_fraction": explained_var
        })

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            comp_df.to_excel(writer, sheet_name="components", index=False)
            weights_df.to_excel(writer, sheet_name="x_weights", index=True)
            loadings_df.to_excel(writer, sheet_name="x_loadings", index=True)
            coef_df.to_excel(writer, sheet_name="coef", index=True)
            r2_df.to_excel(writer, sheet_name="r2_summary", index=True)
            explained_df.to_excel(writer, sheet_name="explained_X", index=False)

    return results


def print_summary(results):
    """Print concise summary of key stats from results dict."""
    print("PLS Summary")
    print("----------")
    print(f"Components: {results['X_scores'].shape[1]}")
    print("R2 per response:")
    for k, v in results["r2"].items():
        print(f" - {k}: {v:.4f}")
    print("Explained variance (X) per component:")
    for i, ev in enumerate(results["explained_var_X"]):
        print(f" - Comp_{i+1}: {ev:.3f}")


if __name__ == "__main__":
    
    # 1. Update paths as needed
    FILE = r"C:\Users\franc\Desktop\tesi2\Data\historical inputs.xlsx"
    SHEET = "Inputs" 
    DATE_COL = "Unnamed: 0" 

    # 2. Mapped Y Targets 
    Y_COLS = [
        "OIS - Wholesale - Weighted average rate - 12 months (bucket) ", 
        "OIS - Wholesale - Weighted average rate - 5 years (bucket) ", 
        "OIS - Wholesale - Weighted average rate - over 10 months (bucket) "
    ] 

    # 3. Mapped X Predictors
    X_COLS = [
        "Crude oil, Brent",
        "Natural gas, Europe",
        "Coal, Australian",
        "Euro nominal effective exchange rate against broad group of trading partners (EER-40), nominal",
        "AAA yield curve - 10-year spot rate (YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y) - Modified value (Monthly)",
        "Air emissions accounts for greenhouse gases by NACE Rev. 2 activity - quarterly data [env_ac_aigg_q__custom_21567527] ------All NACE activities plus households",
        "Eur / Usd",
        "Gross domestic product at market prices, volume",
        "HICP - Overall index",
        "European Union Emissions Trading System-- Primary market",
        "Unemployment rate, age 15 to 74, total",
        "Temperature 2m",
        "Total precipitation",
    ]

    out_file = r"C:/Users/franc/Desktop/tesi2/pls_output.xlsx"

    # Execute the PLS function
    res = run_pls(FILE, SHEET, DATE_COL, X_COLS, Y_COLS, n_components=3, output_path=out_file)
    print(f"Saved: {out_file}")
    print_summary(res)