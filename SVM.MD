# SVM

## Introduction

Support Vector Machines (SVM) is a powerful and versatile supervised machine learning algorithm that is used for classification and regression tasks. It is particularly effective in high-dimensional spaces and is widely employed in various fields, including image classification, handwriting recognition, bioinformatics, and more.

Here's a brief introduction to SVM:

### Definition:
Support Vector Machines are a class of algorithms designed for both classification and regression tasks. They belong to the family of supervised learning algorithms, meaning they learn from labeled training data.

### Objective:
The primary objective of SVM is to find a hyperplane that best separates the data into different classes while maximizing the margin between the classes. The margin is defined as the distance between the hyperplane and the nearest data point of either class. SVM aims to find the hyperplane that not only separates the classes but also maximizes this margin, providing better generalization to new, unseen data.

### Hyperplane:
In a two-dimensional space, a hyperplane is simply a line. In higher dimensions, it becomes a hyperplane. For a binary classification problem, the hyperplane separates the data into two classes, with one class on each side.

### Support Vectors:
Support vectors are the data points that lie closest to the hyperplane and play a crucial role in determining the optimal hyperplane. These are the instances that are most relevant to the decision boundary.

### Kernel Trick:
SVM can efficiently handle non-linear decision boundaries through the use of the kernel trick. Kernels allow SVM to implicitly map the input data into a higher-dimensional space, making it possible to find a linear hyperplane in that transformed space.

### C Parameter:
The C parameter in SVM controls the trade-off between achieving a smooth decision boundary and classifying the training points correctly. A smaller C value allows for a more flexible decision boundary but may misclassify some training points, while a larger C value leads to a stricter decision boundary.

### SVM for Regression:
In addition to classification, SVM can be adapted for regression tasks. In regression, the objective is to fit a hyperplane that captures the trend of the data while minimizing deviations.

### Advantages:
- Effective in high-dimensional spaces.
- Robust against overfitting, especially in high-dimensional space.
- Versatile due to the ability to use different kernel functions.

### Limitations:
- Sensitive to the choice of kernel and its parameters.
- Memory-intensive for large datasets.
- Interpretability can be challenging in complex models.

In summary, SVM is a powerful algorithm that has proven to be effective in various machine learning applications, providing robust and accurate results when properly configured.

## History

The history of Support Vector Machines (SVM) can be traced back to the 1960s and 1970s, with the development of the perceptron and the idea of finding a hyperplane to separate classes in a feature space. The foundations of SVM were laid out over the years through the works of several researchers. Here's a brief timeline highlighting some key developments:

1. **Early Work on Perceptrons (1950s - 1960s):**
   - The concept of a perceptron, a simple algorithm for binary classification, was introduced by Frank Rosenblatt in 1957. A perceptron can be seen as an elementary form of a linear classifier.

2. **Vapnik and Chervonenkis Inequality (1960s):**
   - In the 1960s, Vladimir Vapnik and Alexey Chervonenkis developed the theory of statistical learning, which laid the groundwork for the development of Support Vector Machines. They introduced the Vapnik-Chervonenkis (VC) dimension, which measures the capacity of a learning algorithm to fit a wide range of functions.

3. **Linear Separability and Maximal Margin (1970s):**
   - In 1975, Vapnik introduced the concept of the maximal margin hyperplane. The idea was to find a hyperplane that maximally separates classes while minimizing the risk of misclassification. This led to the development of the support vector classifier.

4. **Introduction of SVM (1992):**
   - The term "Support Vector Machine" was coined by Bernhard Boser, Isabelle Guyon, and Vladimir Vapnik in 1992. They presented the algorithm as an extension of the support vector classifier and introduced the concept of the kernel trick to handle non-linearly separable data.

5. **Kernel Trick (1990s):**
   - The introduction of the kernel trick by Bernhard Boser, Isabelle Guyon, and Vladimir Vapnik allowed SVM to efficiently handle non-linear decision boundaries by implicitly mapping data into a higher-dimensional space. Common kernels include polynomial kernels and radial basis function (RBF) kernels.

6. **Widespread Adoption (1990s - 2000s):**
   - SVM gained popularity in the 1990s and 2000s, becoming a widely used machine learning algorithm for both classification and regression tasks. Its effectiveness in high-dimensional spaces, robustness against overfitting, and versatility contributed to its widespread adoption in various fields.

7. **Sequential Minimal Optimization (SMO) Algorithm (1998):**
   - John Platt proposed the Sequential Minimal Optimization (SMO) algorithm in 1998, providing an efficient way to train SVMs. SMO helped overcome some of the computational challenges associated with solving the optimization problem in SVM training.

The development of SVM represents a convergence of ideas from statistical learning theory, optimization, and geometric concepts. Over the years, SVM has become a fundamental tool in the machine learning toolbox, with applications spanning various domains.

## Application

Support Vector Machines (SVM) find applications in a wide range of fields due to their effectiveness in both classification and regression tasks. Here are some common applications of SVM:

1. **Image Classification:**
   - SVMs are widely used for image classification tasks, such as object recognition and facial recognition. They can effectively learn to distinguish between different objects or faces in images.

2. **Handwriting Recognition:**
   - SVMs are used in handwriting recognition systems to classify handwritten characters. They can learn patterns and features to accurately recognize and classify characters.

3. **Bioinformatics:**
   - SVMs are applied in bioinformatics for tasks such as protein structure prediction, gene expression classification, and disease classification based on genetic data.

4. **Text Classification:**
   - SVMs are commonly used for text classification tasks, such as spam detection, sentiment analysis, and document categorization. They can effectively classify text data into different categories.

5. **Speech Recognition:**
   - SVMs play a role in speech recognition systems, helping in the classification of spoken words or phrases. They can learn to recognize patterns in audio data.

6. **Medical Diagnosis:**
   - SVMs are applied in medical diagnosis for tasks like classifying medical images (e.g., MRI or CT scans), predicting disease outcomes, and identifying patterns in medical data.

7. **Financial Forecasting:**
   - SVMs are used in finance for tasks such as stock market prediction, credit scoring, and fraud detection. They can analyze historical data and identify patterns to make predictions.

8. **Remote Sensing:**
   - In remote sensing applications, SVMs can be used for tasks like land cover classification and object detection in satellite images.

9. **Face Recognition:**
   - SVMs are employed in face recognition systems to classify and identify individuals based on facial features. They can be trained to distinguish between different faces.

10. **Quality Control:**
    - SVMs can be used in manufacturing for quality control purposes, where they can classify products as either defective or non-defective based on various features.

11. **Network Intrusion Detection:**
    - SVMs are utilized in cybersecurity for intrusion detection. They can learn patterns of normal network behavior and detect anomalies indicative of a potential security breach.

12. **Human Activity Recognition:**
    - SVMs find applications in recognizing human activities, such as gesture recognition or activity monitoring. They can classify different activities based on sensor data.

13. **Chemoinformatics:**
    - In chemistry and drug discovery, SVMs can be used for tasks such as predicting the biological activity of chemical compounds or classifying molecules.

The versatility and effectiveness of SVM make it a popular choice in many domains where accurate classification or regression is essential. The ability to handle high-dimensional data and the flexibility provided by kernel functions contribute to SVM's success in various applications.

## Hard Margin SVM

Support Vector Machines (SVMs) are a type of supervised learning algorithm used for classification and regression tasks. The goal of SVMs is to find a hyperplane that best separates the data into different classes. In the case of a hard margin SVM, the algorithm aims to find the hyperplane that maximally separates the data points of different classes while allowing no margin violations.

Here's a detailed explanation of the key concepts in a hard margin SVM:

1. **Linear Separation:**
   - The SVM assumes that the input data can be separated into two classes by a hyperplane. A hyperplane is a decision boundary that divides the feature space into two regions corresponding to different classes.

2. **Margin:**
   - The margin is the distance between the hyperplane and the nearest data point from either class. In a hard margin SVM, the goal is to find the hyperplane that maximizes this margin.

3. **Support Vectors:**
   - Support vectors are the data points that are closest to the hyperplane and have the potential to influence its position. These are the critical elements that determine the location and orientation of the hyperplane.

4. **Decision Function:**
   - The decision function of the SVM is used to classify new data points. Given a set of weights and biases learned during the training process, the decision function assigns a new data point to one of the two classes based on its position relative to the hyperplane.

5. **Optimization Objective:**
   - The objective of a hard margin SVM is to find the optimal hyperplane by maximizing the margin while ensuring that all data points are correctly classified. This is formulated as a constrained optimization problem.

6. **Mathematical Formulation:**
   - The optimization problem can be formulated using the following mathematical expressions:
     - **Objective Function:** Maximize the margin (minimize the norm of the weight vector).
     - **Constraints:** Ensure that each data point is on the correct side of the hyperplane.

7. **Hard Margin vs. Soft Margin:**
   - In a hard margin SVM, the goal is to find a hyperplane with the maximum margin, and it assumes that the data is perfectly separable. However, this assumption may not hold in real-world scenarios. Soft margin SVMs relax this assumption and allow for some margin violations to handle noisy or overlapping data.

8. **Kernel Trick:**
   - SVMs can be extended to handle non-linearly separable data by using the kernel trick. This involves mapping the input data into a higher-dimensional space, where a hyperplane can be more easily found.

In summary, a hard margin SVM seeks to find the optimal hyperplane that maximizes the margin between classes without allowing any data points to be misclassified. This approach is effective when the data is well-behaved and perfectly separable. However, in many real-world cases where the data is not linearly separable or contains noise, a soft margin SVM might be more appropriate.

## Mathematical formulation of Hard Margin SVM

Let's go through the mathematical formulation of a hard margin SVM.

### Notation:

Certainly! Another approach is to use Unicode characters for subscripts and superscripts. This can be more visually appealing and is widely supported:


- w: Weight vector.
- b: Bias term.
- xᵢ: Data point.
- yᵢ: Class label (yᵢ ∈ {-1, 1}).
- wᵀ: Transpose of the weight vector w.
- ξᵢ: Slack variable associated with the i-th data point, allowing for margin violations.


In this notation, Unicode characters like ᵢ, ᵀ, and ᵢ are used for subscripts and superscripts. This method is clean and generally renders well across various Markdown platforms. Keep in mind that the appearance may vary depending on the font and Markdown renderer used.


### Optimization Problem:
The objective is to maximize the margin while ensuring that all data points are correctly classified. This leads to the following optimization problem:

$$
\begin{align*}
&\text{Maximize} \quad \frac{1}{\|\mathbf{w}\|} \\
&\text{Subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i = 1, \ldots, N
\end{align*}
$$

where \(N\) is the number of data points.

### Constraints:
The constraints ensure that each data point is on the correct side of the hyperplane.
The term $$y_i(\mathbf{w}^T \mathbf{x}_i + b)$$ represents the signed distance of the \(i\)-th point to the hyperplane.
For a correctly classified point, this distance is greater than or equal to 1.

### Lagrangian Formulation:
To solve this optimization problem, we introduce Lagrange multipliers (alpha_i) for each constraint:

$$
L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^N \alpha_i \left(y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1\right)
$$

$$\boldsymbol{\alpha} = [\alpha_1, \alpha_2, \ldots, \alpha_N]\$$

### KKT Conditions:

The Karush-Kuhn-Tucker (KKT) conditions for this problem are:

1. first condition:

$$
\frac{\partial L}{\partial \mathbf{w}} = 0
$$

- This gives:

$$
\mathbf{w} = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i
$$

2. second condition:

$$
\frac{\partial L}{\partial b} = 0
$$

- This gives:

$$
\sum_{i=1}^N \alpha_i y_i = 0
$$

3. Non-negativity of Lagrange multipliers

$$
\alpha_i \geq 0
$$

4. The complementary slackness condition.

$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1 \geq 0
$$

### Dual Formulation:

Substitute (W) and (b) back into the Lagrangian to get the dual form:

$$
W(\boldsymbol{\alpha}) = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j
$$

### Optimal W and b:

The optimal weight vector W is obtained as

$$\mathbf{w} = \sum_{i=1}^N \alpha_i y_i \mathbf{x}_i$$

b is determined using any support vector with

$$1 - y_s (\mathbf{w}^T \mathbf{x}_s + b) = 0$$

### Decision Function:
The decision function for a new data point X is given by:

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

### Summary:
1. Solve the dual optimization problem to find optimal Lagrange multipliers (α).
2. Compute W and b using the support vectors.
3. Use the decision function for classification.

This summarizes the mathematical formulation and solution process for a hard margin SVM.

## Limitation of Hard Margin SVM

While hard margin Support Vector Machines (SVMs) are powerful classifiers, they have some limitations that make them less suitable for certain types of datasets:

1. **Sensitivity to Outliers:**
   - Hard margin SVMs are sensitive to outliers because they aim to find a hyperplane with the maximum margin while classifying all training points correctly. Outliers, especially those on the wrong side of the decision boundary, can have a significant impact on the position and orientation of the hyperplane.

2. **Assumption of Linear Separability:**
   - Hard margin SVMs assume that the data is perfectly linearly separable. In real-world scenarios, finding a hyperplane that perfectly separates the classes may not be possible due to noise, measurement errors, or intrinsic data properties. This assumption can lead to overfitting if not satisfied.

3. **Difficulty with Non-Linear Data:**
   - Hard margin SVMs work well when the data is linearly separable, but they struggle with non-linearly separable data. In such cases, a more flexible model, such as a soft margin SVM or a kernelized SVM, might be more appropriate.

4. **Computational Complexity:**
   - The optimization problem for hard margin SVMs involves solving a quadratic programming problem, and its time complexity can be high, especially for large datasets. The training time may become impractical for datasets with a large number of features or instances.

5. **Noisy Data and Overfitting:**
   - Hard margin SVMs are highly sensitive to noise in the data. The attempt to perfectly classify all points can lead to overfitting, especially when dealing with noisy or mislabeled instances. Soft margin SVMs, which allow for some margin violations, are more robust in the presence of noise.

6. **Lack of Probabilistic Outputs:**
   - Hard margin SVMs do not naturally provide probability estimates for class assignments. If probability estimates are required, additional techniques, such as Platt scaling or cross-validation, need to be applied, making the approach less straightforward compared to models that inherently provide probabilities.

7. **Choice of Kernel and Parameters:**
   - When using kernelized SVMs to handle non-linear data, the choice of kernel function and tuning of hyperparameters become crucial. It may require expertise or a time-consuming search process to determine the optimal combination for a given dataset.

To address some of these limitations, soft margin SVMs were introduced. Soft margin SVMs allow for some degree of misclassification and are more robust in the presence of outliers and noise. Additionally, kernelized SVMs provide a way to handle non-linear data by implicitly mapping the input space into a higher-dimensional feature space.

## Hard Margin SVM for Multiclass Classification

1. Basic Concept of SVM:

Support Vector Machine is a supervised machine learning algorithm used for classification and regression analysis. In the case of classification, SVM finds a hyperplane that best separates the data into classes.

2. Hard Margin SVM:

In a binary classification problem, a hard-margin SVM aims to find a hyperplane that maximally separates the two classes and has no data points between the classes. It enforces strict margin constraints, meaning that it allows no misclassifications.

3. Extension to Multiclass Classification:

For a multiclass classification problem, where there are more than two classes, there are different strategies to extend SVM. One common approach is the "One-vs-One" (OvO) or "One-vs-Rest" (OvR) strategy.

   - **One-vs-One (OvO):** Build a binary classifier for every pair of classes. In the end, the class that wins the most pairwise comparisons is the overall winner.

   - **One-vs-Rest (OvR):** Build a binary classifier for each class against the rest. The class with the highest decision function value is assigned.

4. Decision Function:

The decision function of an SVM provides a way to make predictions based on the learned model. It assigns a score to each class, and the class with the highest score is the predicted class.

5. `decision_function_shape` Parameter:

This parameter determines the shape of the decision function returned by the SVM. There are two options:

   - **'ovr' (One-vs-Rest):** The decision function has shape (n_samples, n_classes). Each row corresponds to a sample, and each column corresponds to a class. This is consistent with the decision functions returned by other classifiers.

   - **'ovo' (One-vs-One):** The decision function has shape (n_samples, n_classes * (n_classes - 1) / 2). This is the original format used by LIBSVM. Each row corresponds to a sample, and each column corresponds to a pair of classes.

6. Changes and Recommendations:

   - In scikit-learn version 0.17, the default value for `decision_function_shape` was set to 'ovr'.
   - As of version 0.19, 'ovr' is the default, and 'ovo' is deprecated.
   - In version 0.17, using 'ovr' is recommended.
   - Internally, scikit-learn uses the 'ovo' strategy for training models, but 'ovr' is constructed from the 'ovo' matrix when needed.

This change in default values and recommendations reflects the practicality and efficiency of the 'ovr' strategy for multiclass classification in most cases. It simplifies the decision function and aligns with common practices in machine learning libraries.