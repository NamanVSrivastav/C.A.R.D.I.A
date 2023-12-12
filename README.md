# C.A.R.D.I.A
C.A.R.D.I.A = CArdiovascular Risk  prediction using DIstributed  A.I.


C.A.R.D.I.A is a project aimed at predicting cardiovascular risk through advanced technologies like Federated Learning (Distributed Ai) and Feedforward Neural Networks. This repository contains the codebase and documentation for the project.




---

**Client-Side Script: Federated Learning Client for Heart Disease Prediction**

The client.py script serves as a client in a Federated Learning (FL) system designed to predict heart disease. 
It leverages the Flower (FLwr) framework to facilitate decentralized model training. 
Key components include a Keras Feedforward Neural Network (FFNN) model, a custom `CifarClient` class for interfacing with Flower, and functions for model training and evaluation. 
The client loads a subset of the heart disease dataset, updates its local FFNN model during training rounds, and contributes to the collaborative learning process. This client script is an integral part of a larger FL system and is intended to run on decentralized devices. The overarching goal is collectively improving the FFNN model's accuracy while preserving data privacy.

---

**Server-Side Script: Federated Learning Server for Heart Disease Prediction**

The server.py script represents the server-side implementation of a Federated Learning (FL) system for predicting heart disease. 
Utilizing the Flower (FLwr) framework, the script orchestrates federated training across multiple clients. 
It defines a Keras Feedforward Neural Network (FFNN) model, configures the Federated Averaging (FedAvg) strategy, and sets up evaluation and configuration functions. The server script collaborates with FL clients, aggregating their local updates to improve the FFNN model's performance over multiple rounds. 
The federated learning process includes dynamically adjusting training and evaluation configurations. Additionally, the script visualizes the evolution of loss and accuracy across rounds.
This server-side script is a fundamental component of a distributed FL system, fostering collaborative learning for heart disease prediction.


---

**Instructions**
:

**Step 1:** Navigate to the folder containing your files and open the Command Prompt (cmd).

**Step 2:** Install the necessary tools by executing the command "pip install -r requirements.txt" in the Command Prompt.

After confirming the installation of the requirements, please go ahead and initiate the server and client scripts.

**Step 3:** Open another Command Prompt in the same folder as your files.

**Step 4:** Activate the server script by typing and executing "python server.py". This ensures the server is operational on the designated port (in this case, port 8080).

**Step 5:** Launch three additional Command Prompt instances (adjust the number based on the minimum client count set in server.py).

**Step 6:** Run "python client.py" in each Command Prompt instance.

Once all instances are executing the code, view the final results in the server.py Command Prompt. The outcomes of each round and the ultimate result will be showcased in the server.py instance.


---
**Accuracy achieved**: 89.98% 
---
**Dataset used**: https://www.kaggle.com/datasets/zhaoyingzhu/heartcsv
---
**Conclusion**:

In this work, a significant advancement in cardiovascular risk prediction by combining the capabilities of artificial intelligence (AI) with the privacy-preserving 
features of FL are addressed. 
The proposed FL model not only attained high accuracy rates but also exhibited adaptability to various deployment scenarios. 
In addition, it also 
addressed existing gaps in predictive techniques and laid the groundwork for personalized and confidential healthcare interventions. 
The promising results highlighted the potential integration of AI in healthcare, particularly in the early detection and prevention of cardiovascular diseases. 
The proposed model can preserve the privacy of healthcare data and attain an accuracy of up to 89.9%.

---
**Future work** may explore further enhancements to the FL model, considering diverse datasets and expanding its scope to 
encompass predictions for a broader range of medical conditions, thereby contributing 
to the ongoing evolution of predictive healthcare technologies.

---
