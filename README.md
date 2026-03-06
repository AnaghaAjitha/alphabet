# alphabet

dataset
https://drive.google.com/drive/folders/1iOU06_3BEb3g37N8l6-ui-j5aimhC-Vu?usp=sharing



audio capture
        |
mel spectrogram conversion
(frequency representation of speech)
        |
       CNN
extract local frequency patterns
detects energy bursts
learns time–frequency structures
        |
   BiLSTM Layer
track temporal evolution of features
learns phoneme transitions
captures speech dynamics
        |
fully connected Layer
        |
     softmax 
        |
letter predicted
