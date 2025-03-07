# ai-workshop-2025-minigrad

## Description

This repository, `ai-workshop-2025-minigrad`, is a step-by-step educational project designed to build a simple automatic differentiation engine (minigrad) from scratch, inspired by Andrej Karpathy's micrograd.  It's aimed at teaching the fundamental concepts behind autodifferentiation, backpropagation, and neural networks. The project includes a `Value` object that tracks operations and gradients, along with implementations of basic neural network components (Neuron, Layer, MLP). The code and explanation is structured in a Jupyter Notebooks for an interactive learning.

The project emphasizes understanding the chain rule, topological sorting for backpropagation, and implementing various mathematical operations (addition, multiplication, power, tanh, exp, relu).  It culminates in building a small multi-layer perceptron (MLP) using the custom-built engine.

## Dependencies

This project has the following dependencies, as specified in `pyproject.toml`:

*   **numpy** (>=2.2.3,<3.0.0): Numerical computing library.
*   **torch** (>=2.6.0,<3.0.0): Used for comparison and demonstration of a full-featured autodiff engine.
*   **matplotlib** (>=3.10.1,<4.0.0):  Used for visualizations (though primarily graphviz is used).
*   **graphviz** (>=0.20.3,<0.21.0): Used to visualize the computational graph.
* **jupyter** (>=1.1.1,<2.0.0): Notebooks and related tools.

You also need to have `graphviz` installed at the system level (not just the Python package). This is typically done via your system's package manager (e.g., `apt install graphviz` on Debian/Ubuntu).

The project also clones itself within the notebook, this ensures it is always the latest version.

## Installation and Setup

There are several ways to run this, via Jupyter, Google Colab, or as Python scripts.

### 1. Using Jupyter Locally

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Antonio-Tresol/ai_workshop_2025_minigrad.git
    cd ai_workshop_2025_minigrad
    ```

2.  **Install Dependencies (using Poetry):**

    It's highly recommended to use a virtual environment.  The `pyproject.toml` file uses Poetry for dependency management. If you have Poetry installed, simply run:

    ```bash
    poetry install
    ```

    If you don't have poetry. Install it:
     ```bash
    pip install poetry
    ```

    This will create a virtual environment and install all the necessary packages.

3. **Install Graphviz system-wide:**
    On Debian/Ubuntu:
    ```bash
        sudo apt update
        sudo apt install graphviz
    ```
    On Fedora/CentOS/RHEL:
    ```bash
        sudo yum install graphviz
    ```
    On macOS (using Homebrew):
    ```bash
        brew install graphviz
    ```
    On Windows, download the installer from the official Graphviz website.

4.  **Launch Jupyter Notebook:**

    ```bash
    poetry run jupyter notebook
    ```

    This will open the Jupyter Notebook interface in your web browser. Navigate to and open either `blank.ipynb` or `step_by_step.ipynb`.

5.  **Run the Notebook:**

    Execute the cells in the notebook sequentially. The notebook contains both code and explanations.

### 2. Using Google Colab

The `blank.ipynb` notebook includes a "Open in Colab" badge. Clicking this badge will open the notebook directly in Google Colab.  Colab provides a pre-configured environment, so you *don't* need to install dependencies manually.  However, the notebook *does* include cells to clone the repository and install dependencies (including system-level graphviz) within the Colab environment.  This approach ensures you're always using the latest version of the code.

### 3. Running as Python Scripts (Advanced)

While the project is primarily designed to be used with Jupyter Notebooks, the core logic is contained within the `engine` directory:

*   `engine/engine.py`:  Contains the `Value` class, which is the fundamental building block of the autodiff engine.  It implements the forward and backward passes for various operations.
*   `engine/nn.py`: Implements basic neural network components (Neuron, Layer, MLP) using the `Value` class.

You *could* technically import and use these modules directly in your own Python scripts.  However, the notebooks provide much richer context and explanations.  This method is only recommended for advanced users who are already comfortable with the concepts.

## Project Structure

*   **`blank.ipynb`**: A Jupyter Notebook with the code and explanations, providing an interactive learning experience, designed to be filled in by the user.
*   **`step_by_step.ipynb`**: (Not provided in the original input, but implied) A Jupyter Notebook similar to `blank.ipynb`, but with the complete, filled-in code.
*   **`engine/`**: Contains the core logic of the autodiff engine.
    *   `engine/engine.py`:  The `Value` class and its methods.
    *   `engine/nn.py`: Neural network components.
*   **`img/`**: (Empty in the provided structure)  Presumably intended for images used in the notebooks or documentation.
*   **`pyproject.toml`**: Defines project metadata and dependencies (using Poetry).
*   **`LICENSE`**: The MIT License file.

## How to Build (Development)

This section is relevant if you intend to modify the code or contribute to the project. Since the project is built and tested in the notebooks, build commands are just for quality and standards.

1.  **Clone and Install:** Follow steps 1-3 in the "Using Jupyter Locally" section above.

2.  **Development:**  Make changes to the code in `engine/engine.py` or `engine/nn.py`.  The notebooks will automatically use the updated code when you re-run the cells.

3.  **Testing:** The notebooks themselves serve as the primary testing mechanism. Ensure that all cells execute without errors and that the outputs are as expected.

## Key Concepts

*   **Automatic Differentiation (Autodiff):**  The core concept being taught. Autodiff is a technique for automatically computing derivatives of functions, which is crucial for training neural networks.
*   **Computational Graph:**  The `Value` objects and their relationships form a computational graph that represents the mathematical expression being evaluated.
*   **Forward Pass:**  The process of evaluating the expression, starting from the inputs and propagating values through the graph.
*   **Backward Pass (Backpropagation):**  The process of computing gradients, starting from the output and propagating them back through the graph using the chain rule.
*   **Chain Rule:**  The fundamental mathematical rule used for backpropagation.  The `_backward` methods in `engine.py` implement the local chain rule for each operation.
*   **Topological Sort:**  Used to ensure that the `_backward` methods are called in the correct order during backpropagation.
*   **Neural Network Components:**  The `Neuron`, `Layer`, and `MLP` classes demonstrate how to build basic neural networks using the `Value` class.

## Contributing

Contributions are welcome!  If you find bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Copyright by Andrej Karpathy and Antonio Badilla-Olivas.
