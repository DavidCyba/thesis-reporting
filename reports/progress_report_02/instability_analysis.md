---
layout: default
title: Instability Analysis
---


# Instability Analysis

Ich habe hier ein paar unterschiedliche Architekturen immer für jeweils die Konfiguration von 

$$m = 2.0$$ 
$$V=4^3\times 8$$ 
$$LR = 1 \cdot 10^{-3}$$ 
geplottet. Die Instabilitäten scheinen dabei allerdings mit der Masse und der Anzahl an Layern zu korrelieren. So erhalte ich z.B für nur eine Layer nie Instabilitäten und für eine Masse von $m = 0.0$ erst Instabilitäten für mehr als 10 Layer.

<br>

(DW bezeichnet folgend den Dirac-Wilson Operator und DWC den Dirac-Wilson Clover Operator.
Pathtype 2 bezeichnet folgend die One-Hop-Paths und Pathtype 3 die Circular Paths nach:)

<br>

    def path_generator(type: int):

        zero_hop_paths = [[]]
        one_hop_paths = [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

        circular_paths = []

        for mu in range(4):
            for nu in range(4):
                if nu == mu:
                    continue
                circular_paths.append([(mu, 1), (nu, 1), (mu, -1), (nu, -1)])
                circular_paths.append([(mu, 1), (nu, -1), (mu, -1), (nu, 1)])
                circular_paths.append([(mu, -1), (nu, 1), (mu, 1), (nu, -1)])
                circular_paths.append([(mu, -1), (nu, -1), (mu, 1), (nu, 1)])

        match type:
            case 1:
                return zero_hop_paths
            case 2:
                return zero_hop_paths + one_hop_paths
            case 3:
                return zero_hop_paths + one_hop_paths + circular_paths

<br>

##### DW, 4 Layers, Pathtype 2, Identity Initialization

<br>

![One-Hop DW vs DWC](./images/DW%20-%20Using%20HCRes_VarAdapt%204%20layers%20with%207000%20steps%20and%20pathtype%202_M2.0_Identitiy.png)

<br>

Diese Konfiguration sollte eigentlich die Werte aus dem Paper reproduzieren. Hier werden auch immernoch die One-Hop-Paths verwendet. Obwohl die Minima der Spikes zwar weiter konvergieren, lässt sich auch hier erkennen, dass diese Architektur nicht die Werte aus dem Paper in einem Zeitraum unter 10000 Training Steps erreichen kann.

<br>

##### DW, 4 Layers, Pathtype 2, Identity Initialization, LR Update

<br>

![One-Hop DW vs DWC](./images/DW%20-%20Using%20HCRes_VarAdapt%204%20layers%20with%207000%20steps%20and%20pathtype%202_M2.0_Identity_LR.png)

<br>

Hier wurde manuell die Learning Rate ab dem Auftreten von Instabilitäten angepasst. Es lässt sich erkennen, dass dadurch die Instabilitäten vermieden werden, in manchen Fällen konnte so sogar noch ein besserer Wert für die Approximation Quality erzielt werden wie ohne die Benutzung von Scheduling. Allerdings kann auch so nicht der gewünschte Wert von ungefähr $6 \cdot 10^{-5}$ für eine Masse von $m = 2.0$ für eine Architektur von **4 Layer, DW, HC Model** aus dem Paper erzielt werden.

<br>


![One-Hop DW vs DWC](./images/DW%20-%20Using%20HCRes_VarAdapt%204%20layers%20with%207000%20steps%20and%20pathtype%202_M2.0_Hopping.png)

<br>

Hier wurde die Initialisierung mit der Hopping Expansion bei sonst identischer Konfiguration übernommen. Eine eigens geschriebene Implementierung dieser Initialisierung steht noch aus. Hier zeigt sich allerdings, dass so ohne Instabilitäten der Wert von ungefähr $6 \cdot 10^{-5}$ aus dem Paper erzielt werden kann.

<br>

Folgend sind noch weitere Beispielplots für verschiedene Architekturen abgebildet:

<br>

##### DWC, 16 Layers, Pathtype 2, Identity Initialization

![One-Hop DW vs DWC](./images/DWC%20-%20Using%20HCRes_VarAdapt%2016%20layers%20with%207000%20steps%20and%20pathtype%202_M2.0.png)

##### DWC, 32 Layers, Pathtype 2, Identity Initialization

![One-Hop DW vs DWC](./images/DWC%20-%20Using%20HCRes_VarAdapt%2032%20layers%20with%204000%20steps%20and%20pathtype%202_M2.0.png)

<br>

Hier zeigt sich, dass überraschenderweise für den DWC Operator das Training mit der Identity Initialisierung trotz Masse von $m = 2.0$ und einer großen Anzahl and Layer stabil verläuft.

<br>

##### DWC, 16 Layers, Pathtype 3, Identity Initialization

![One-Hop DW vs DWC](./images/DWC%20-%20Using%20HCRes_VarAdapt%2016%20layers%20with%204000%20steps%20and%20pathtype%203_M2.0.png)

<br>

##### Fazit

Das Auftreten der Instabilitäten scheint sowohl für die One-Hop-Paths als auch für die Circular Paths stattzufinden. Hierbei kann eine lose Korrelation mit größeren Massen und der Tiefe des Netzwerks festgestellt werden. Learning Rate Scheduling schafft es zwar die Instabilitäten zu minimieren, nur mit der Hopping Expansion können allerdings die Werte aus dem Paper reproduziert werden.

<br>

Die Hopping Expansion müsste ich noch selber implementieren, da ich aktuell nur eine Kopie zum Troubleshooten verwendet habe. Hierbei bedarf es noch etwas überlegung wie ich diese mit der verschiedenen Anzahl an Pfaden in den PT Layers kompatibel mache.


