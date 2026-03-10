# Proiect Autonomous Driving Perception

Autonomous Driving Perception - Semantic Segmentation

Acest proiect implementează un sistem bazat pe cunoaștere dedicat segmentării semantice a scenariilor de trafic urban, o componentă critică pentru vehiculele autonome. Utilizând arhitectura U-Net, modelul clasifică fiecare pixel din imaginile capturate de camera frontală în 13 categorii distincte (drum, vehicule, pietoni, vegetație etc.).

Caracteristici Cheie

    Dataset Complex: Utilizarea setului de date CARLA Densely Annotated Driving Dataset, compus din peste 10.700 de cadre din 28 de secvențe video.

    Robustete Meteo: Antrenare și evaluare pe scenarii diverse: însorit, noros și ploaie (cu reflexii și distorsiuni vizuale).

    Split Stratificat: Strategie de divizare a datelor bazată pe metadate (vreme + trafic), nu doar pe cadre aleatorii, pentru o evaluare onestă a capacității de generalizare.

    Optimizare Performanță: Implementare cu Automatic Mixed Precision (AMP) pentru accelerarea antrenării pe GPU și Weighted Cross-Entropy Loss pentru a combate dezechilibrul claselor (ex: pietoni vs. drum).

Arhitectura Modelului (U-Net)

Modelul folosește o structură simetrică formată din:

    Encoder (Downsampling): Extrage caracteristici abstracte complexe prin convoluții și max-pooling.

    Decoder (Upsampling): Reconstruiește rezoluția imaginii pentru localizarea precisă a obiectelor.

    Skip Connections: Transferă informația de înaltă rezoluție de la encoder la decoder, esențial pentru detectarea obiectelor mici, precum semnele de circulație.

Tehnologii Utilizate

    Language: Python 

    Deep Learning Framework: PyTorch 

    Hardware Acceleration: NVIDIA CUDA & AMP 

    Data Handling: Pandas & NumPy
