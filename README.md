# alphabet

dataset
https://drive.google.com/drive/folders/1IyLXsN025Pn7KEXjSWLdwz5NeDP8AAQg?usp=sharing



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
