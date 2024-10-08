import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random


st.set_page_config(layout="wide")

# Define the patterns (zero_array, one_array, etc.) here
zero_array=np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
                    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                    [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

one_array=np.array([[-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1]])

two_array=np.array([[1, 1, 1, 1, 1, 1, 1, 1, -1, -1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1], 
                    [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1], 
                    [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
                    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
                    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1], 
                    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1], 
                    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
                    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1]])

three_array=np.array([[-1, -1, 1, 1, 1, 1, 1, 1, -1, -1], 
                    [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1], 
                    [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1], 
                    [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1], 
                    [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1], 
                    [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
                    [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1], 
                    [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1]])

four_array=np.array([[-1, 1, 1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
                     [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
                     [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
                     [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1], 
                     [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1]])

six_array=np.array([[1, 1, 1, 1, 1, 1, -1, -1, -1, -1], 
                    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1], 
                    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1], 
                    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1], 
                    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1], 
                    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1], 
                    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1], 
                    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1], 
                    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1],
                    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1], 
                    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1]])

square_array=np.array([[1, 1, 1, 1, 1, -1, -1, -1, -1, -1], 
                       [1, 1, 1, 1, 1, -1, -1, -1, -1, -1], 
                       [1, 1, 1, 1, 1, -1, -1, -1, -1, -1], 
                       [1, 1, 1, 1, 1, -1, -1, -1, -1, -1], 
                       [1, 1, 1, 1, 1, -1, -1, -1, -1, -1], 
                       [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

nine_array=np.array([[-1, -1, -1, -1, 1, 1, 1, 1, 1, 1], 
                     [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
                    [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1], 
                    [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                    [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                    [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                    [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1], 
                    [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], 
                    [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], 
                    [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1], 
                    [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1]])


input_array = np.array([zero_array, one_array, two_array, three_array, four_array, six_array, square_array, nine_array])

def create_weight_matrix(input_arrays):
    weights = np.zeros((120, 120))
    for i in range(120):
        for j in range(120):
            if i == j:
                weights[i, j] = 0
            else:
                w = 0
                for l in range(len(input_arrays)):
                    w += input_arrays[l, i] * input_arrays[l, j]
                weights[i, j] = w / input_arrays.shape[0]
                weights[j, i] = weights[i, j]
    return weights

def add_noise(pattern, noise_level):
    noisy_pattern = pattern.copy()
    num_pixels = int(120 * noise_level)
    for _ in range(num_pixels):
        pixel = random.randint(0, 119)
        noisy_pattern[pixel] *= -1
    return noisy_pattern

def synchronous_update(pattern, weights):
    return np.sign(np.dot(weights, pattern))

def asynchronous_update(pattern, weights, max_iterations):
    n = len(pattern)
    updated_pattern = pattern.copy()
    for _ in range(max_iterations):
        i = random.randint(0, n-1) 
        activation = np.dot(weights[i], updated_pattern)
        new_state = 1 if activation > 0 else -1 if activation < 0 else updated_pattern[i]
        updated_pattern[i] = new_state
    return updated_pattern


def calculate_energy_evolution(pattern, weights, update_mode, iterations):
    energy_list = []
    current_pattern = pattern.copy()

    for _ in range(iterations):
        energy = -0.5 * np.dot(np.dot(current_pattern, weights), current_pattern)
        energy_list.append(energy)

        if update_mode == "Synchronous":
            current_pattern = synchronous_update(current_pattern, weights)
        else:
            current_pattern = asynchronous_update(current_pattern, weights, 1)

    return energy_list


def main():
    st.title("Hopfield Network Visualization 🧠️")

    st.sidebar.header("Network Settings")
    selected_number = st.sidebar.selectbox("Select a number", ["0", "1", "2", "3", "4", "6", "9", "Square"])
    noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.25, 0.01)
    update_mode = st.sidebar.radio("Update mode", ["Synchronous", "Asynchronous"])
    iterations = st.sidebar.slider("Iterations (for Asynchronous)", 1, 500, 5)

    input_arrays = input_array.reshape(8, 120)
    weights = create_weight_matrix(input_arrays)

    pattern_index = ["0", "1", "2", "3", "4", "6", "Square", "9"].index(selected_number)
    original_pattern = input_arrays[pattern_index]

    noisy_pattern = add_noise(original_pattern, noise_level)

    if update_mode == "Synchronous":
        recovered_pattern = synchronous_update(noisy_pattern, weights)
    else:
        recovered_pattern = asynchronous_update(noisy_pattern, weights, iterations)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(original_pattern.reshape(12, 10), cmap='gray', interpolation='nearest')
    ax1.set_title("Original Pattern")
    ax1.axis('off')

    ax2.imshow(noisy_pattern.reshape(12, 10), cmap='gray', interpolation='nearest')
    ax2.set_title(f"Noisy Pattern ({noise_level:.2%} noise)")
    ax2.axis('off')

    ax3.imshow(recovered_pattern.reshape(12, 10), cmap='gray', interpolation='nearest')
    ax3.set_title(f"Recovered Pattern ({update_mode})")
    ax3.axis('off')

    st.pyplot(fig)

    st.subheader("Hopfield Network Information")
    st.write(f"Update Mode: {update_mode}")
    if update_mode == "Asynchronous":
        st.write(f"Number of Iterations: {iterations}")
    st.write(f"Noise Level: {noise_level:.2%}")

    accuracy = np.mean(np.abs(original_pattern - recovered_pattern) < 0.1) * 100
    st.write(f"Recovery Accuracy: {accuracy:.2f}%")

    if st.button("Show Energy Evolution"):
        energy_evolution = calculate_energy_evolution(noisy_pattern, weights, update_mode, iterations)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(energy_evolution)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.set_title("Energy Evolution during Pattern Recovery")
        st.pyplot(fig)

if __name__ == "__main__":
    main()