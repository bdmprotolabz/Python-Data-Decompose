
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
from flask import Flask, render_template, request, flash, session, redirect, url_for,jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense,Dropout # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau #type: ignore
 
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.losses import MeanSquaredError
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from sklearn.metrics import mean_squared_log_error
from tensorflow.keras.optimizers import Adam #type: ignore

# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB file size limit

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Handle file size limit error
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash('File size exceeds the allowed limit of 100 MB.')
    app.logger.warning("Attempted file upload exceeds limit.")
    return redirect(url_for('index'))

# Allowed file extensions check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)

                # Read file based on extension
                if filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding='ISO-8859-1')  # Fallback encoding
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath)

                # Preprocess the data
                df_cleaned = preprocess_data(df)
                print('dropna method data', df_cleaned)
                df_normalized = apply_min_max_normalization(df_cleaned)   
                # Apply Min-Max normalization
                print('data frame after apply min max', df_normalized)
                
                fetaured_column,correlation_results = perform_correlation(df_normalized, target_column='windmill_generated_power(kW/h)')
                
                
                
                print('correlation result', correlation_results)
                print('featured result', fetaured_column)
        
                if correlation_results.empty:
                    flash('No correlation found for "windmill_generated_power(kW/h)"')
                    return redirect(url_for('index'))
                

                # Send all numeric columns and correlation results to the template
                session['columns'] = df_cleaned.select_dtypes(include=['number']).columns.tolist()
                session['correlation_results'] = correlation_results.to_dict()  # Convert to dictionary
                session['featured_column'] = fetaured_column.columns.to_list()
                session['datafile_path'] = filepath  # Save file path instead of dataframe

                flash('File uploaded and analyzed. Highly correlated columns selected with threshold > 0.5. Contains only positive related columns.')
                return redirect(url_for('index'))

            except pd.errors.EmptyDataError:
                flash("Uploaded file is empty or invalid.")
            except pd.errors.ParserError:
                flash("Error parsing the file. Ensure the file format is correct.")
            except Exception as e:
                flash(f"Error reading file: {e}")
                app.logger.error(f"Error occurred while processing the file: {e}", exc_info=True)
                return redirect(request.url)

    
    
    columns = session.get('columns',[])
    correlation_data = session.get('correlation_results',{})
    return render_template('index.html', columns = columns, correlation_data = correlation_data)



@app.route('/select_variables', methods=['POST'])
def select_variables():
    
    selected_columns = request.form.getlist('columns')
    decomposition_method = request.form.get('decomposition')  # Get the selected decomposition method
    session['selected_columns'] = selected_columns
    session['decomposition_method'] = decomposition_method  # Store the selected method in the session

    # Capture dynamic parameters based on selected method
    if decomposition_method in ['wavelet_decomposition', 'wavelet_packet_decomposition']:
        wavelet_type = request.form.get('wavelet')  # Get wavelet type
        level = int(request.form.get('level'))  # Get decomposition level
        session['wavelet_type'] = wavelet_type
        session['level'] = level

    elif decomposition_method == 'variational_mode_decomposition':
        alpha = float(request.form.get('alpha'))  # Get alpha value
        tau = float(request.form.get('tau'))  # Get tau value
        K = int(request.form.get('K'))  # Get number of modes (K)
        DC = int(request.form.get('DC'))  # Get DC value (0 or 1)
        init = int(request.form.get('init'))  # Get initialization value
        # tol = float(request.form.get('tol'))  # Get tolerance value

        # Store all VMD parameters in the session
        session['alpha'] = alpha
        session['tau'] = tau
        session['K'] = K
        session['DC'] = DC
        session['init'] = init
        # session['tol'] = tol
        
    
    
    return redirect(url_for('decomposition'))
    # return redirect(url_for('decomposition'))

@app.route('/decomposition')
def decomposition():
    selected_columns = session.get('selected_columns', [])
    selected_col = selected_columns
    decomposition_method = session.get('decomposition_method', None)
    datafile_path = session.get('datafile_path', None)
    featured_column = session.get('featured_column',[])

    print('selected column',selected_columns)
    # Capture and convert additional parameters
    wavelet_type = session.get('wavelet_type', None)
    level = session.get('level', None)
    alpha = session.get('alpha', None)
    
    # Variational Mode Decomposition (VMD) parameters
    tau = session.get('tau', None)
    K = session.get('K', None)
    DC = session.get('DC', None)
    init = session.get('init', None)
    # tol = session.get('tol', None)

    # Convert level and alpha to integers or floats
    if level is not None:
        level = int(level)  # Convert level to an integer
    if alpha is not None:
        alpha = float(alpha)  # Convert alpha to a float
    if tau is not None:
        tau = float(tau)  # Convert tau to a float
    if K is not None:
        K = int(K)  # Convert K (number of modes) to an integer
    if DC is not None:
        DC = int(DC)  # Convert DC to an integer
    if init is not None:
        init = int(init)  # Convert init to an integer
    # if tol is not None:
    #     tol = float(tol)  # Convert tolerance to a float

    # Continue with the rest of your logic...
    if not selected_columns:
        flash('No variables selected for decomposition.')
        return redirect(url_for('index'))

    if decomposition_method is None:
        flash('No decomposition method selected.')
        return redirect(url_for('index'))

    if datafile_path is None:
        flash('No data found for decomposition.')
        return redirect(url_for('index'))

    # Reload the DataFrame from the file
    if datafile_path.endswith('.csv'):
        df = pd.read_csv(datafile_path)
    elif datafile_path.endswith('.xlsx'):
        df = pd.read_excel(datafile_path)

    # Extract the selected columns for decomposition
    selected_columns = df[selected_columns].values
    featured_column = df[featured_column].values

    print('data of selected column', featured_column)
    
    # Perform the decomposition
    if decomposition_method == 'variational_mode_decomposition':
        metrics, plot_urls = perform_decomposition(
            selected_columns, featured_column, decomposition_method, alpha=alpha, tau=tau, K=K, DC=DC, init=init
        )
    else:
        # Handle wavelet-based decompositions
        metrics, plot_urls = perform_decomposition(
            selected_columns, featured_column, decomposition_method, wavelet_type=wavelet_type, level=level
        )
    return render_template(
    'decomposition.html',
    selected_col=selected_col,
    decomposition_method=decomposition_method,
    metrics=metrics,  # Ensure this contains 'mse', 'mae', and 'rmse' if used in the template
    plot_urls=[f"{plot}" for plot in plot_urls]  # Adjust relative to 'static'
)




# def perform_decomposition(data, method, wavelet_type=None, level=None, alpha=None, tau=None, K=None, DC=None, init=None, tol=None):
#     print('data Comming for Cnn Prediction', data)
#     # Setting random seeds for reproducibility
#     np.random.seed(42)
#     tf.random.set_seed(42)
#     random.seed(42) 

#     tau = float(tau) if tau is not None else 0.0
#     K = int(K) if K is not None else 3
#     DC = int(DC) if DC is not None else 0
#     init = int(init) if init is not None else 1
#     tol = 1e-6

#     # Ensure data is 1-dimensional
#     if len(data.shape) > 1:
#         data = data.flatten()

#     # Handle NaN values by replacing them with the mean of the data
#     if np.isnan(data).any():
#         data_mean = np.nanmean(data)
#         data = np.where(np.isnan(data), data_mean, data)

#     # Normalize the data (scaling between 0 and 1)
#     data_min, data_max = np.min(data), np.max(data)
#     if data_min == data_max:
#         data_normalized = np.zeros_like(data)
#     else:
#         data_normalized = (data - data_min) / (data_max - data_min)

#     # Decompose the data based on the selected method
#     if method == 'wavelet_decomposition':
#         import pywt
#         coeffs = pywt.wavedec(data_normalized, wavelet_type, level=level)
#         mid_level = level - 1
#         decomposition_result = coeffs[mid_level]  # Approximation coefficients

#     elif method == 'wavelet_packet_decomposition':
#         import pywt
#         wp = pywt.WaveletPacket(data_normalized, wavelet_type, maxlevel=level)
#         decomposition_result = wp['a' * level].data  # Approximation coefficients at the selected level

#     elif method == 'emd':
#         from PyEMD import EMD
#         emd = EMD()
#         IMFs = emd.emd(data_normalized)
#         decomposition_result = IMFs[0]  # First IMF
        
#     elif method == 'ceemd':
#         from PyEMD import CEEMDAN
#         ceemdan = CEEMDAN()
#         IMFs = ceemdan.ceemdan(data_normalized)
        
#         # Choose specific IMFs or use all of them
#         decomposition_result = IMFs  # This gives you all the IMFs
#     elif method == 'variational_mode_decomposition':
#         import vmdpy
#         decomposition_result, _, _ = vmdpy.VMD(data_normalized, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
#         decomposition_result = decomposition_result[0]  # First decomposed mode

#     else:
#         raise ValueError("Unknown decomposition method")

#     # Prepare the data for CNN prediction (reshaping into 3D array for CNN)
#     X = np.arange(len(decomposition_result)).reshape(-1, 1)  # Time steps
#     y = decomposition_result  # The decomposed result as target

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Reshape the input data for CNN (samples, timesteps, features)
#     X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#     # Build a revised CNN for regression
#     cnn_model = Sequential()
#     cnn_model.add(Input(shape=(X_train_cnn.shape[1], 1)))

#     # Convolutional layers
#     cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
#     if X_train_cnn.shape[1] > 1:
#         cnn_model.add(MaxPooling1D(pool_size=2))  # Only use pooling if time_steps > 1
#     else:
#         print("Warning: Not enough time steps for pooling. Skipping MaxPooling1D.")

#     cnn_model.add(Dropout(0.5, seed=42))

#     cnn_model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))

#     if X_train_cnn.shape[1] > 1:
#         cnn_model.add(MaxPooling1D(pool_size=2))  # Only use pooling if time_steps > 1
#     else:
#         print("Warning: Not enough time steps for pooling. Skipping MaxPooling1D.")

#     cnn_model.add(Dropout(0.5, seed=42))

#     # Flatten the output
#     cnn_model.add(Flatten())

#     # Fully connected layers
#     cnn_model.add(Dense(100, activation='relu'))
#     cnn_model.add(Dense(1))  # Output layer for regression

#     # Compile the model
#     cnn_model.compile(optimizer='adam', loss='mean_squared_error')

#     # Define the learning rate reduction callback
#     lr_reduction = ReduceLROnPlateau(
#         monitor='val_loss',    # Monitor the validation loss
#         factor=0.5,            # Reduce the learning rate by 50%
#         patience=5,            # Wait for 5 epochs before reducing the learning rate
#         min_lr=1e-6            # Set a minimum learning rate
#     )

#     # Train the model with callbacks
#     cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=50, batch_size=32, verbose=1, callbacks=[lr_reduction], shuffle=False)

#     # Make predictions using CNN
#     y_pred_cnn = cnn_model.predict(X_test_cnn)

#     # Calculate evaluation metrics
#     mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
#     rmse_cnn = np.sqrt(mean_squared_error(y_test, y_pred_cnn))
#     epsilon = 1e-10
#     mape_cnn = np.mean(np.abs((y_test - y_pred_cnn.flatten()) / (y_test + epsilon))) * 100
#     mse = mean_squared_error(y_test, y_pred_cnn)

#     # Plot the original data vs CNN prediction
#     plt.figure(figsize=(10, 6))
#     plt.plot(X_test, y_test, label='True Data', marker='o')
#     plt.plot(X_test, y_pred_cnn, label='CNN Predicted Data', linestyle='--', marker='x')
#     plt.title(f'CNN Prediction after {method.capitalize()}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     # Save the plot to a BytesIO object and encode it in base64
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
#     plt.close()

#     # Return evaluation metrics and the plot
#     metrics_result_cnn = f"""
#     CNN MAE: {mae_cnn:.4f}
#     CNN RMSE: {rmse_cnn:.4f}
#     CNN MSE: {mse:.4f}
#     """

#     return metrics_result_cnn, f"data:image/png;base64,{plot_url}"
import pywt
from PyEMD import EMD
from vmdpy import VMD  # Example VMD implementation
import os

def perform_decomposition(
    selected_column, featured_columns, method,
    wavelet_type=None, level=None, alpha=None, tau=None,
    K=None, DC=None, init=None, tol=None
):
    """
    Perform decomposition on the selected column, use featured columns for CNN prediction,
    and evaluate CNN predictions for each decomposition level.

    Returns:
    - metrics: A dictionary of CNN evaluation metrics for each decomposition level.
    - plot_urls: A list of file paths (URLs) to the saved decomposition plots.
    """
    # Debug input shapes
    print("SelectedColumn Shape:", selected_column.shape)
    print("FeaturedColumns Shape:", featured_columns.shape)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    tau = float(tau) if tau is not None else 0.0
    K = int(K) if K is not None else 3
    DC = int(DC) if DC is not None else 0
    init = int(init) if init is not None else 1
    tol = 1e-6
    # Flatten selected_column if needed
    if len(selected_column.shape) > 1:
        selected_column = selected_column.flatten()
    print("Flattened selected_column shape:", selected_column.shape)

    # Handle NaN values in selected_column
    if np.isnan(selected_column).any():
        print("NaN values found in selected_column. Replacing with column mean.")
        data_mean = np.nanmean(selected_column)
        selected_column = np.where(np.isnan(selected_column), data_mean, selected_column)

    # Normalize selected_column
    data_min, data_max = np.min(selected_column), np.max(selected_column)
    if data_min == data_max:
        print("Warning: selected_column has constant value.")
        selected_normalized = np.zeros_like(selected_column)
    else:
        selected_normalized = (selected_column - data_min) / (data_max - data_min)
    print("Normalized selected_column (first 10 values):", selected_normalized[:10])

    # Initialize variables
    decomposition_result = None
    plot_urls = []
    output_dir = os.path.join("static", "plots")
    os.makedirs(output_dir, exist_ok=True)
    # Perform decomposition based on method
    if method == 'wavelet_decomposition':
        import pywt
        coeffs = pywt.wavedec(selected_normalized, wavelet_type, level=level)
        decomposition_result = coeffs
        print(f"Wavelet Decomposition produced {len(coeffs)} levels.")

        # Save decomposition plots
        for i, coeff in enumerate(decomposition_result):
            plt.figure(figsize=(10, 4))
            plt.plot(coeff)
            plt.title(f'Level {i + 1} Decomposition')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'decomposition_level_{i + 1}.png')
            plt.savefig(plot_path)
            plt.close()
            plot_urls.append(plot_path)

    elif method == 'emd':
        from PyEMD import EMD
        emd = EMD()
        
        # Perform EMD decomposition
        decomposition_result = emd.emd(selected_normalized)
        
        # Limit the number of IMFs to 3
        max_imfs = 3
        decomposition_result = decomposition_result[:max_imfs]
        print(f"EMD produced {len(decomposition_result)} IMFs.")

        # Save IMF plots
        for i, imf in enumerate(decomposition_result):
            plt.figure(figsize=(10, 4))
            plt.plot(imf)
            plt.title(f'IMF {i + 1}')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'imf_{i + 1}.png')
            plt.savefig(plot_path)
            plt.close()
            plot_urls.append(plot_path)

    elif method == 'variational_mode_decomposition':
        import vmdpy
        decomposition_result, _, _ = vmdpy.VMD(
            selected_normalized, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol
        )
        print(f"VMD produced {len(decomposition_result)} modes.")

        # Save mode plots
        for i, mode in enumerate(decomposition_result):
            plt.figure(figsize=(10, 4))
            plt.plot(mode)
            plt.title(f'Mode {i + 1}')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'mode_{i + 1}.png')
            plt.savefig(plot_path)
            plt.close()
            plot_urls.append(plot_path)

    else:
        raise ValueError("Unknown decomposition method")

    # Check decomposition result
    print("Decomposition result lengths:", [len(level) for level in decomposition_result])
    if any(len(level) == 0 for level in decomposition_result):
        raise ValueError("Empty decomposition levels found.")

    # Dictionary to store CNN metrics
    metrics = {}

    for i, level_data in enumerate(decomposition_result):
        print(f"Training CNN for decomposition level {i + 1}...")

        # Align lengths of X (features) and y (decomposed target)
        min_length = min(featured_columns.shape[0], len(level_data))
        X = featured_columns[:min_length]
        y = level_data[:min_length]

        print(f"Aligned X shape: {X.shape}, Aligned y length: {len(y)}")


        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print(f"NaN values detected in X or y for level {i + 1}.")
            valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]

        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            print(f"Inf values detected in X or y for level {i + 1}.")
            valid_indices = ~np.isinf(X).any(axis=1) & ~np.isinf(y)
            X = X[valid_indices]
            y = y[valid_indices]
        
        if len(y) == 0:
            raise ValueError(f"Decomposition level {i + 1} resulted in empty target data.")

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Normalize features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Normalize target (optional but recommended)
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        print(f"y_train_scaled (first 5): {y_train_scaled[:5]}")
        print(f"y_test_scaled (first 5): {y_test_scaled[:5]}")

        # Reshape for CNN input
        X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
        X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

        print(f"X_train_cnn shape: {X_train_cnn.shape}")
        print(f"X_test_cnn shape: {X_test_cnn.shape}")

        # Check for NaN or Inf after scaling
        if np.any(np.isnan(X_train_cnn)) or np.any(np.isnan(y_train_scaled)):
            raise ValueError("NaN values found in scaled X_train or y_train.")
        if np.any(np.isinf(X_train_cnn)) or np.any(np.isinf(y_train_scaled)):
            raise ValueError("Infinite values found in scaled X_train or y_train.")

        # Build CNN model
        cnn_model = Sequential([
            Input(shape=(X_train_cnn.shape[1], 1)),
            Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal'),
            MaxPooling1D(pool_size=2) if X_train_cnn.shape[1] > 1 else Dropout(0.5),
            Dropout(0.5),
            Conv1D(filters=128, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal'),
            MaxPooling1D(pool_size=2) if X_train_cnn.shape[1] > 1 else Dropout(0.5),
            Dropout(0.5),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(1)  # Output for regression
        ])
        cnn_model.compile(
            optimizer=Adam(learning_rate=0.0001, clipnorm=1.0),  # Lower learning rate and clip gradients
            loss='mean_squared_error'
        )

        # Train the model
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        history = cnn_model.fit(
            X_train_cnn, y_train_scaled,
            validation_data=(X_test_cnn, y_test_scaled),
            epochs=50, batch_size=32,
            callbacks=[lr_reduction],
            verbose=1
        )

        # Predictions
        y_pred_cnn = cnn_model.predict(X_test_cnn).flatten()

        # Rescale predictions back to original scale
        y_pred_rescaled = y_scaler.inverse_transform(y_pred_cnn.reshape(-1, 1)).flatten()
        y_test_rescaled = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

        print(f"Sample y_test_rescaled: {y_test_rescaled[:5]}")
        print(f"Sample y_pred_rescaled: {y_pred_rescaled[:5]}")

        # Metrics
        mae_cnn = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        rmse_cnn = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
        mse_cnn = mean_squared_error(y_test_rescaled, y_pred_rescaled)

        metrics[f'Level {i + 1}'] = {
            'MAE': mae_cnn,
            'RMSE': rmse_cnn,
            'MSE': mse_cnn
        }

        print(f"Metrics for Level {i + 1}: MAE={mae_cnn}, RMSE={rmse_cnn}, MSE={mse_cnn}")

    return metrics, plot_urls


import glob
IMAGE_FOLDER = "static/plots"
@app.route('/delete-images', methods=['POST'])
def delete_images():
    try:
        # Find all image files in the folder (e.g., jpg, png, gif)
        image_files = glob.glob(os.path.join(IMAGE_FOLDER, '*'))
        
        # Delete each file
        for file_path in image_files:
            if os.path.isfile(file_path):  # Ensure it's a file before deleting
                os.remove(file_path)
        
        return jsonify({'message': 'All images deleted successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def wavelet_decomposition(data, wavelet_type, level):
    coeffs = pywt.wavedec(data, wavelet=wavelet_type, level=level)
    return coeffs  # Decomposition levels


def wavelet_packet_decomposition(data, wavelet_type, level):
    wp = pywt.WaveletPacket(data, wavelet=wavelet_type, mode='symmetric', maxlevel=level)
    return [node.data for node in wp.get_level(level, order='natural')]


from PyEMD import EMD, EEMD, CEEMDAN

def emd_decomposition(data, method):
    if method == 'emd':
        emd = EMD()
        return emd(data)
    elif method == 'eemd':
        eemd = EEMD()
        return eemd(data)
    elif method == 'ceemdan':
        ceemdan = CEEMDAN()
        return ceemdan(data)


def variational_mode_decomposition(data, alpha, tau, K, DC, init):
    modes, _ = VMD(data, alpha, tau, K, DC, init)
    return modes  # Decomposed modes


def cnn_prediction(decomposition_level, featured_column):
    # Ensure decomposition_level is reshaped correctly for Conv1D (samples, time_steps, features)
    decomposition_level = np.expand_dims(decomposition_level, axis=-1)  # Adding feature dimension (e.g., (samples, time_steps, 1))
    
    # Example CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(decomposition_level.shape[1], 1)),  # Adjust kernel size to 1
        tf.keras.layers.MaxPooling1D(pool_size=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Reshape for CNN
    X = decomposition_level
    y = featured_column

    # Train/Test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train the model
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Predict and calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mse, mae, rmse





import os

def plot_decomposition(data, title):
    # Ensure the temp directory exists
    os.makedirs('temp', exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)

    # Save plot as image and return URL (use temporary directory for production)
    plot_path = f'temp/{title}.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path



def preprocess_data(df):
    # Optionally: Add more preprocessing steps like scaling, outlier removal, etc.
    df_cleaned = df.dropna()  # Simple method, you can expand this
    return df_cleaned

# def perform_correlation(df):
#     numerical_columns = df.select_dtypes(include=['number']).columns
#     if numerical_columns.empty:
#         flash("No numerical columns available for correlation analysis.")
#         return pd.DataFrame()

#     correlation_matrix = df[numerical_columns].corr(method='pearson')
    
#     print("Correlation Matrix" , correlation_matrix)


# 


#     return correlation_matrix


# def perform_correlation(df, target_column='System power generated | (kW)'):
    
#     print("Data passed to perform_correlation (df):")
#     print(df.head())  # Print the data passed to the function to verify it's normalized
#     # Select numeric columns
#     numerical_columns = df.select_dtypes(include=['number']).columns
#     # Ensure that the target column exists in the dataframe
#     if target_column not in numerical_columns:
#         flash(f"Target column {target_column} not found in the dataset.")
#         return pd.DataFrame()

#     # pd.options.display.max_columns = 9999``
#     print("System row",pd.options.display.max_columns)
#     # Calculate the correlation matrix
#     correlation_matrix = df[numerical_columns].corr(method='pearson')
#     print('correlation Matrix',correlation_matrix)

#     # Get correlations for the target column with all other numeric columns
#     correlation_with_target = correlation_matrix[target_column].drop(target_column)  # Exclude self-correlation

#     # Return a dictionary of correlation values for the target column
#     return correlation_with_target

def perform_correlation(df, target_column='windmill_generated_power(kW/h)', threshold=0.06):
    print("Data passed to perform_correlation (df):")
    print(df.head())  # Print the data passed to the function to verify it's normalized
    
    # Select numeric columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    # Ensure that the target column exists in the dataframe
    if target_column not in numerical_columns:
        flash(f"Target column {target_column} not found in the dataset.")
        return pd.DataFrame()

    # Calculate the correlation matrix
    correlation_matrix = df[numerical_columns].corr(method='pearson')
    print('Correlation Matrix:', correlation_matrix)

    # Get correlations for the target column with all other numeric columns
    correlation_with_target = correlation_matrix[target_column].drop(target_column)  # Exclude self-correlation
    print(f"Correlation with target {target_column}:\n", correlation_with_target)

    # Filter columns based on the threshold
    selected_columns = correlation_with_target[correlation_with_target.abs() > threshold].index.tolist()
    print(f"Selected columns with correlation > {threshold}:", selected_columns)

    # Return a dataframe with the selected features and the target column
    features_column = df[selected_columns]
   
    return features_column, correlation_with_target


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def filter_columns_by_correlation(correlation_matrix, threshold=0.2):
    included_columns = []  # To store columns that meet the threshold
    max_included_value = []  # To store max correlation values for included columns
    excluded_columns = []  # To store columns that don't meet the threshold (only column names)
    excluded_max_value = []  # To store max correlation for excluded columns
    excluded_values_dict = {}  # Dictionary to store excluded values per column

    for column in correlation_matrix.columns:
        # Find all positive correlations excluding self-correlation
        positive_correlations = correlation_matrix[column][(correlation_matrix[column] > 0) & (correlation_matrix[column] < 1)]

        # Find the max positive correlation
        max_correlation = positive_correlations.max()

        # Separate excluded correlations (those below threshold)
        excluded_values = positive_correlations[positive_correlations <= threshold].tolist()

        if max_correlation and max_correlation > threshold:
            # Include the column with its max correlation and excluded values
            included_columns.append(column)
            max_included_value.append(max_correlation)
            
            excluded_values_dict[column] = excluded_values
        else:
            # Exclude the column by just appending its name
            excluded_columns.append(column)
            excluded_max_value.append(max_correlation)
            # excluded_values_dict[column] = excluded_values

    return included_columns, max_included_value, excluded_values_dict, excluded_columns, excluded_max_value

# def filter_columns_by_correlation(correlation_matrix, threshold=0.2):
#     included_columns = []  # To store columns that meet the threshold for positive correlation
#     max_positive_values = []  # To store max positive correlation values for included columns
#     max_negative_values = []  # To store max negative correlation values for included columns
#     excluded_columns = []  # To store columns that don't meet the threshold (only column names)
#     excluded_max_value = []  # To store max correlation for excluded columns
#     excluded_values_dict = {}  # Dictionary to store excluded values per column

#     target_column = correlation_matrix.columns[0]  # Assuming the first column is the target
    
#     # Loop through the correlation values of all other columns with the target
#     for index, value in correlation_matrix[target_column].items():
#         if index == target_column:
#             continue  # Skip self-correlation
#          # Ensure the correlation value is a float or int
#         if not isinstance(value, (int, float)):
#             continue  # Skip non-numerical values
#         # Get the positive and negative correlations
#         positive_correlation = value if value > 0 else 0
#         negative_correlation = value if value < 0 else 0
        
#         # Handle positive correlation (above threshold)
#         if positive_correlation > threshold:
#             included_columns.append(index)
#             max_positive_values.append(positive_correlation)
#         else:
#             excluded_columns.append(index)
#             excluded_max_value.append(positive_correlation)
        
#         # Handle negative correlation (if it exists)
#         if negative_correlation < -threshold:
#             excluded_columns.append(index)
#             excluded_max_value.append(negative_correlation)
        
#         # Store excluded values in the dictionary
#         excluded_values_dict[index] = {
#             'positive_correlation': positive_correlation,
#             'negative_correlation': negative_correlation
#         }

#     return included_columns, max_positive_values, excluded_values_dict, excluded_columns, excluded_max_value





@app.route('/clear_session', methods=['POST'])
def clear_session():
    # Clear specific session keys or all session data
    session.pop('correlation_results', None)  # Clear the selected columns from session
    session.pop('columns',None)
    # session.pop('excluded_values_dict',None)
    # session.pop('excluded_columns',None)
    # session.pop('max_excluded_value',None)

    # You can also clear other session data if needed
    return '', 204  # Return a successful response with no content


def apply_min_max_normalization(df):
    numerical_columns = df.select_dtypes(include=['number']).columns  # Only normalize numerical columns
    scaler = MinMaxScaler()  # Initialize the scaler
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])  # Fit and transform the data
    return df

@app.template_filter('rounding')
def rounding(value, precision=2):
    return round(value, precision)

if __name__ == '__main__':
    app.run(debug=True)
