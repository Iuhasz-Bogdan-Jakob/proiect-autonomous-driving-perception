# Proiect Autonomous Driving Perception

Autonomous Driving Perception - Semantic Segmentation

Acest proiect implementează un sistem bazat pe cunoaștere dedicat segmentării semantice a scenariilor de trafic urban, o componentă critică pentru vehiculele autonome. Utilizând arhitectura U-Net, modelul clasifică fiecare pixel din imaginile capturate de camera frontală în 13 categorii distincte (drum, vehicule, pietoni, vegetație etc.). A fost folosită tehnica de split stratificat ca modelul să fie capabil să învețe din toate situațile de trafic și să fie pregătit pentru o situație reală. De asemenea, s-a implementat și AMP (Automatic Mixed Precision for Deep Learning) pentru a reduce timpul de antrenare pe setul de date și weight-loss entropy care penalizează modelul grav dacă greșește clasele rare precum pietoni (clasă critică). Modelul a fost antrenat pe un set de date de aproximativ 10.700 de imagini din simulatorul CARLA care a permis generarea tuturor situaților de trafic care pot apărea.





