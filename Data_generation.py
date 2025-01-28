import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import os


label_mapping = joblib.load('models/label_mapping.pkl')
print("Label Mapping loaded:", label_mapping)

#------------------------(Directories)------------------------
scaler_folder_path = 'models/scaler_weights'
model_folder_path = 'models/gan_models'

#------------------------(Directories)--------------------------

def generate_synthetic_data(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, generator.input_shape[1]))
    synthetic_data = generator.predict(noise)
    return synthetic_data

synthetic_data_df = pd.DataFrame()


for attack_type, attack_label in label_mapping.items():
    generator_model_path = os.path.join(model_folder_path, f'generator_{attack_type}.h5')
    generator = tf.keras.models.load_model(generator_model_path)
    print(f"Generator model for {attack_type} loaded")
    
    scaler_path = os.path.join(scaler_folder_path, f'{attack_label}_scaler.pkl')

    scaler = joblib.load(scaler_path)
    print(f"Scaler for {attack_type} loaded")


    num_samples = 50000  # Number of samples to generate per attack type
    synthetic_data = generate_synthetic_data(generator, num_samples)
    
    synthetic_data = scaler.inverse_transform(synthetic_data)

    
    synthetic_data_df_temp = pd.DataFrame(synthetic_data, columns=[f'feature_{i}' for i in range(synthetic_data.shape[1])])
    synthetic_data_df_temp['Attack_type'] = attack_label

    
    synthetic_data_df = pd.concat([synthetic_data_df, synthetic_data_df_temp], ignore_index=True)



synthetic_data_df.to_csv('synthetic_dataset.csv', index=False)
print("Synthetic dataset saved as 'synthetic_dataset.csv'")
