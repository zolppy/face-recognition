# Celebrity Face Recognition

This project demonstrates how to build and train a Convolutional Neural Network (CNN) for face recognition. 
The dataset is pulled from Kaggle via `kagglehub`, preprocessed with OpenCV, and used to train a deep learning model in TensorFlow/Keras. 
The notebook walks through dataset loading, preprocessing, model building, training, evaluation, and visualization of predictions.

---

## 📂 Project Structure

├── main.ipynb # Jupyter Notebook with code, training, and evaluation

├── README.md # Project documentation


---

## 📊 Dataset

The dataset is downloaded using [`kagglehub`](https://pypi.org/project/kagglehub/).  
It contains labeled images across multiple categories. Images are preprocessed using **OpenCV** and split into training and testing sets.

---

## 🧠 Model Architecture

The CNN is implemented using **TensorFlow/Keras** with the following structure:

- **Conv2D** + **MaxPooling2D** layers for feature extraction  
- **Flatten** layer to convert feature maps into vectors  
- **Dense layers** with ReLU activations  
- **Softmax output layer** for classification  

The model is compiled with:

- Optimizer: `Adam`  
- Loss: `categorical_crossentropy`  
- Metrics: `accuracy`

---

## ⚙️ Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/yourusername/celebrity-face-recognition.git
cd celebrity-face-recognition
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- kagglehub

You can install them manually with:

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn kagglehub
```

---

## 🚀 Usage

Run the notebook step by step:

```
jupyter notebook main.ipynb
```

The notebook includes:

- Dataset download & preprocessing
- Model training
- Evaluation & accuracy results
- Visualization of sample predictions

---

## 📈 Results
The model achieves competitive accuracy on the test dataset.
Example prediction visualization:
- Input image shown with label
- Model prediction displayed with confidence
(See `main.ipynb` for detailed plots and accuracy results.)

---

## 📌 Future Work
- Improve performance with data augmentation
- Experiment with deeper CNN architectures
- Deploy the trained model with a web interface

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to modify.

---

## 📜 License
This project is licensed under the MIT License.
