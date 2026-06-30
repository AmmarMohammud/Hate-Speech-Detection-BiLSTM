# Automated Hate Speech & Offensive Language Detection System

An end-to-end Deep Learning project designed to automatically detect and classify text (tweets/comments) into three categories: **Hate Speech**, **Offensive Language**, or **Neither (Clean)**. Built using **TensorFlow/Keras**, this project leverages advanced NLP architectures to address toxic online behavior.
======================================================================================================================================================================================================================================================================================================================
## 🚀 Key Features
* **Advanced Architecture:** Uses a **Bidirectional LSTM** network to capture contextual information from both preceding and succeeding text sequences.
* **Optimized Data Pipeline:** Implements the `tf.data` API featuring memory caching, shuffling, and prefetching to ensure efficient memory utilization and high-speed training.
* **Interactive Notebook Demo:** Includes a real-time testing interface built with `ipywidgets` right inside the Jupyter Notebook for instant predictions.
* **Overfitting Prevention:** Integrated with `EarlyStopping` callbacks and fine-tuned `Dropout` regularizations.
======================================================================================================================================================================================================================================================================================================================
## 📊 Dataset
The model is trained on the widely recognized **Hate Speech and Offensive Language Dataset** (originally hosted on Kaggle by *mrmorj*). It comprises labeled tweets categorized into:
1. `Hate Speech` (Class 0)
2. `Offensive Language` (Class 1)
3. `Neither` (Class 2)
======================================================================================================================================================================================================================================================================================================================
## 🏗️ Model Architecture
The architecture is structured sequentially to process raw text into dense semantic predictions:

+-------------------------------+-----------------------+-----------+
| Layer (Type)                  | Output Shape          | Param #   |
+-------------------------------+-----------------------+-----------+
| Embedding                     | (None, None, 64)      | 640,064   |
| Bidirectional (LSTM)          | (None, 128)           | 66,048    |
| Dense (ReLU)                  | (None, 128)           | 16,512    |
| Dropout (0.5)                 | (None, 128)           | 0         |
| Dense (ReLU)                  | (None, 64)            | 8,256     |
| Dropout (0.5)                 | (None, 64)            | 0         |
| Dense (Softmax)               | (None, 3)             | 195       |
+-------------------------------+-----------------------+-----------+

* Total params: 731,075 (2.79 MB)
* Trainable params: 731,075 (2.79 MB)
* Non-trainable params: 0 (0.00 Byte)
======================================================================================================================================================================================================================================================================================================================
## 🛠️ Tech Stack & Libraries
* **Framework:** TensorFlow / Keras (v2.x)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Interactive UI:** ipywidgets / IPython.display
* **Environment:** Kaggle Notebooks / Python 3
======================================================================================================================================================================================================================================================================================================================
## 💻 How to Run the Notebook
1. Clone the repository:
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)

2. Install the necessary dependencies if running locally:
   pip install tensorflow pandas numpy matplotlib ipywidgets

3. Open the `.ipynb` file using Jupyter Notebook or upload it to Kaggle/Google Colab and run all cells.
4. Scroll to the bottom to interact with the Live Classification Demo.