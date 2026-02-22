 Hidden Markov Model using Baum–Welch Algorithm

Name: Aneena S S
University Register Number: TCR24CS012

 Description
This project implements a Hidden Markov Model (HMM) trained using the Baum–Welch algorithm.The model learns the transition and emission probabilities from a given observation sequence.

 Input
- Observation sequence
- Number of hidden states

 Output
- Initial distribution (π)
- Transition matrix (A)
- Emission matrix (B)
- Probability P(O | λ) per iteration

 Visualization
The learning process is visualized using a graph showing P(O | λ) versus iteration number.

 How to Run
pip install -r requirements.txt  
python main.py  
python visualize.py
