
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


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


X = data.drop(columns=[attack_column])
y = data[attack_column]

synthetic_data = pd.read_csv('synthetic_dataset.csv')


synthetic_feature_names = X.columns.tolist()
synthetic_data.columns = synthetic_feature_names + [attack_column]


X_synthetic = synthetic_data.drop(columns=[attack_column])
y_synthetic = synthetic_data[attack_column]

X_combined = pd.concat([X, X_synthetic], ignore_index=True)
y_combined = pd.concat([y, y_synthetic], ignore_index=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_classifier_with_mixed_data(X, y, X_combined, y_combined, kf, classifier):
    original_accuracies = []
    combined_accuracies = []
    for train_index, test_index in tqdm(kf.split(X), total=kf.get_n_splits(), desc="K-Fold Progress"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
      
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        original_accuracies.append(accuracy_score(y_test, y_pred))

        classifier.fit(pd.concat([X_train, X_combined], ignore_index=True),
                       pd.concat([y_train, y_combined], ignore_index=True))
        y_combined_pred = classifier.predict(X_test)
        combined_accuracies.append(accuracy_score(y_test, y_combined_pred))

    return np.mean(original_accuracies), np.mean(combined_accuracies)


classifier = RandomForestClassifier(n_estimators=100, random_state=42)


original_accuracy, combined_accuracy = evaluate_classifier_with_mixed_data(X, y, X_combined, y_combined, kf, classifier)
print(f"Accuracy on original data (k-fold): {original_accuracy}")
print(f"Accuracy on original + synthetic data (k-fold): {combined_accuracy}")
