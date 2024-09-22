import pennylane as qml

# def quantum_circuit_angle_entangle(inputs, weights, n_qubits):
#     qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
#     qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
#     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

def quantum_circuit_angle_entangle(inputs, weights, n_qubits):
    # Explicit AngleEmbedding gates (RY rotations)
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Explicit BasicEntanglerLayer gates
    for layer in range(len(weights)):
        for i in range(n_qubits):
            qml.RX(weights[layer][i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])  # Chain entanglement
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]