
import tensorflow as tf
#-----------------------(Ignore if working on non gpu)---------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured GPU: {gpus}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

#-----------------------(Ignore if working on non gpu)---------------
 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os
import time
from tqdm import tqdm



tf.get_logger().setLevel('ERROR')




#-----------------(Pre_processing)-----------------------------------------
data = pd.read_csv('DNN-EdgeIIoT-dataset.csv', low_memory=False)


drop_columns = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
    "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
    "tcp.dstport", "udp.port", "arp.opcode", "mqtt.msg", "icmp.unused",
    "http.tls_port", 'dns.qry.type', 'dns.retransmit_request_in', "mqtt.msg_decoded_as",
    "mbtcp.trans_id", "mbtcp.unit_id", "http.request.method", "http.referer",
    "http.request.version", "dns.qry.name.len", "mqtt.conack.flags",
    "mqtt.protoname", "mqtt.topic"
]
data.drop(drop_columns, axis=1, inplace=True)
data.dropna(inplace=True)

attack_column = 'Attack_type'
label_encoder = LabelEncoder()
data[attack_column] = label_encoder.fit_transform(data[attack_column])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

#---------------------------------(Directories)-------------------------------------
label_mapping_path = './models/label_mapping.pkl'
scaler_folder_path = './models/scaler_weights'
model_folder_path = './models/gan_models'

os.makedirs(os.path.dirname(label_mapping_path), exist_ok=True)
os.makedirs(scaler_folder_path, exist_ok=True)
os.makedirs(model_folder_path, exist_ok=True)

#----------------------------------(Directories)--------------------------------------

joblib.dump(label_mapping, label_mapping_path)
print("Label Mapping saved")
#----------------------------------(Pre_processing)----------------------------------



#-------------------------------(Neural_Networks)---------------------------------

def build_generator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='leaky_relu', input_dim=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='leaky_relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2048, activation='leaky_relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='leaky_relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='leaky_relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
#-----------------------------(Neural_Networks)--------------------------------------



#-----------------------------(Training)------------------------------

with tf.device('/GPU:0'):  #Remove if non Gpu training  
    for attack_type, attack_label in label_mapping.items():
        print(f"Processing Attack Type: {attack_type}")
        
        Current_data = data[data[attack_column] == attack_label]

        X = Current_data.drop(columns=[attack_column])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler
        scaler_path = os.path.join(scaler_folder_path, f'{attack_label}_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler for {attack_type} saved")

        
        input_dim = X_scaled.shape[1]
        generator = build_generator(input_dim)
        discriminator = build_discriminator(input_dim)

    
        gen_optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.Adam(0.00001, beta_1=0.5)
        cross_entropy = tf.keras.losses.BinaryCrossentropy()

        
        epochs = 1000
        batch_size = 3000

    
        generator_losses = []
        discriminator_losses = []
        discriminator_accuracies = []
        generator_accuracies = []

        
        for epoch in tqdm(range(epochs), desc=f'Training GAN for {attack_type}'):
            idx = np.random.randint(0, X_scaled.shape[0], batch_size)
            real_samples = tf.convert_to_tensor(X_scaled[idx], dtype=tf.float32)

            
            noise = tf.random.normal([batch_size, input_dim])
            fake_samples = generator(noise, training=True)

        
            with tf.GradientTape() as disc_tape:
                real_pred = discriminator(real_samples, training=True)
                fake_pred = discriminator(fake_samples, training=True)
                real_loss = cross_entropy(tf.ones_like(real_pred), real_pred)
                fake_loss = cross_entropy(tf.zeros_like(fake_pred), fake_pred)
                disc_loss = real_loss + fake_loss

            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    
            with tf.GradientTape() as gen_tape:
                noise = tf.random.normal([batch_size, input_dim])
                fake_samples = generator(noise, training=True)
                fake_pred = discriminator(fake_samples, training=True)
                gen_loss = cross_entropy(tf.ones_like(fake_pred), fake_pred)

            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

           
            generator_losses.append(gen_loss.numpy())
            discriminator_losses.append(disc_loss.numpy())
            real_accuracy = tf.reduce_mean(tf.cast(real_pred > 0.5, tf.float32)).numpy()
            fake_accuracy = tf.reduce_mean(tf.cast(fake_pred < 0.5, tf.float32)).numpy()
            discriminator_accuracies.append((real_accuracy + fake_accuracy) / 2)
            generator_accuracies.append(1 - fake_accuracy)

        #Accuracy PLot of each type
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(generator_losses, label="Generator Loss")
        plt.plot(discriminator_losses, label="Discriminator Loss")
        plt.legend()
        plt.title(f"GAN Losses for {attack_type}")

        plt.subplot(2, 1, 2)
        plt.plot(discriminator_accuracies, label="Discriminator Accuracy")
        plt.plot(generator_accuracies, label="Generator Accuracy (Disc on Fake)")
        plt.legend()
        plt.title(f"Accuracies for {attack_type}")
        plt.xlabel("Epochs")

        plt.tight_layout()
        plt.show()
#------------------------------(Training)------------------------------------------------
        
        generator.save(os.path.join(model_folder_path, f'generator_{attack_type}.h5'))
        print(f"Model for {attack_type} saved")

    
        time.sleep(5)

print("GAN training for all attack types complete!")
