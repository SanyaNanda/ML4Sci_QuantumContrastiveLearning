import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pennylane as qml

# Data Loading
class DataLoader:
    def __init__(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.pairs_train = np.expand_dims(data["pairs_train"], -1)
        self.labels_train = data["labels_train"]
        self.pairs_test = np.expand_dims(data["pairs_test"], -1)
        self.labels_test = data["labels_test"]
    
    def get_train_data(self):
        return self.pairs_train, self.labels_train
    
    def get_test_data(self):
        return self.pairs_test, self.labels_test

# Define Quantum Circuit Function
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Quantum Circuit Class
class QuantumCircuit:
    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = qml.QNode(quantum_circuit, self.dev)

    def get_quantum_layer(self):
        return qml.qnn.KerasLayer(self.qnode, self.weight_shapes, output_dim=self.n_qubits)

# CNN Model with Quantum Layer
class QuantumCNN:
    def __init__(self, input_shape, quantum_layer, n_qubits=4):
        self.input_shape = input_shape
        self.quantum_layer = quantum_layer
        self.n_qubits = n_qubits
    
    def create_model(self, return_embeddings=False):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu')) # Conv layer 1
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Conv layer 2
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
    
        # ------ Quantum layer added in nb 2 classical base architecture
        # Reducing dimensions to match n_qubits
        model.add(layers.Dense(n_qubits)) 
        # Quantum layer
        model.add(quantum_layer)
        # Dense layer after quantum layer
        model.add(layers.Dense(n_qubits, activation='relu'))
        if return_embeddings:
            return model
        # --------------------------------------------------------------
        
        # model.add(layers.Dense(1, activation='sigmoid'))  
        return model


# Contrastive Loss
class Losses:
    @staticmethod
    def quantum_fidelity_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def infonce_loss(margin=1.0):
        def loss(y_true, dist):
            y_true = tf.cast(y_true, tf.float32)
            square_dist = tf.square(dist)
            margin_square = tf.square(tf.maximum(margin - dist, 0))
            return tf.reduce_mean(y_true * square_dist + (1 - y_true) * margin_square)
        return loss

# Siamese Network
class SiameseNetwork:
    def __init__(self, input_shape, quantum_cnn):
        self.input_shape = input_shape
        self.quantum_cnn = quantum_cnn

    def create_network(self):
        base_model = self.quantum_cnn.create_model()

        input_0 = layers.Input(shape=self.input_shape)
        input_1 = layers.Input(shape=self.input_shape)

        processed_0 = base_model(input_0)
        processed_1 = base_model(input_1)

        distance = layers.Lambda(
            lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=-1)),
            output_shape=(1,)
        )([processed_0, processed_1])

        siamese_model = models.Model([input_0, input_1], distance)
        return siamese_model

# Training
class Trainer:
    def __init__(self, siamese_network, pairs_train, labels_train, pairs_test, labels_test):
        self.siamese_network = siamese_network
        self.pairs_train = pairs_train
        self.labels_train = labels_train
        self.pairs_test = pairs_test
        self.labels_test = labels_test

    def train(self, epochs=10, batch_size=128, learning_rate=1e-3):
        tf.get_logger().setLevel('ERROR')

        self.siamese_network.compile(
            loss=Losses.infonce_loss(),
            optimizer=optimizers.Adam(learning_rate=learning_rate)
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='hybrid_model_qg1_c2.h5', save_weights_only=True, verbose=1)

        history = self.siamese_network.fit(
            [self.pairs_train[:, 0], self.pairs_train[:, 1]], self.labels_train,
            validation_data=([self.pairs_test[:, 0], self.pairs_test[:, 1]], self.labels_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[cp_callback]
        )
        return history

# Main Execution
if __name__ == "__main__":
    data_loader = DataLoader('../../data/quark_gluon_dataset/qg_20000_pairs_c4.npz')
    pairs_train, labels_train = data_loader.get_train_data()
    pairs_test, labels_test = data_loader.get_test_data()

    quantum_circuit = QuantumCircuit()
    quantum_layer = quantum_circuit.get_quantum_layer()

    quantum_cnn = QuantumCNN(pairs_train.shape[2:], quantum_layer)
    siamese_network = SiameseNetwork(pairs_train.shape[2:], quantum_cnn).create_network()

    trainer = Trainer(siamese_network, pairs_train, labels_train, pairs_test, labels_test)
    trainer.train(epochs=10, batch_size=128)
