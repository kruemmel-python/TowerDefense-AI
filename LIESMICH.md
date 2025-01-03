# Tower Defense Reinforcement Learning

Dieses Repository enthält eine Implementierung von Reinforcement Learning für ein Tower Defense Spiel unter Verwendung von Deep Q-Networks (DQN) mit priorisiertem Experience Replay. Die Spielumgebung und der Agent werden mit Python, Pygame, NumPy und TensorFlow erstellt.

## Inhaltsverzeichnis

- [Einführung](#einführung)
- [Funktionen](#funktionen)
- [Installation](#installation)
- [Nutzung](#nutzung)
- [Konfiguration](#konfiguration)
- [Training](#training)
- [Ergebnisse](#ergebnisse)
- [Mitwirkung](#mitwirkung)
- [Lizenz](#lizenz)

## Einführung
![image](https://github.com/user-attachments/assets/23e364d8-b297-42d9-96a4-dd23c6a1a83a)

Tower Defense ist ein klassisches Strategiespiel, bei dem der Spieler gegen Wellen von Feinden verteidigen muss, indem er strategisch Verteidigungstürme platziert. Dieses Projekt implementiert einen Reinforcement Learning Agenten, der lernt, das Spiel effektiv zu spielen, indem er Türme platziert und aufrüstet sowie spezielle Fähigkeiten einsetzt.

## Funktionen

- **Tower Defense Umgebung**: Eine benutzerdefinierte Umgebung, die mit Pygame erstellt wurde.
- **Deep Q-Network (DQN) Agent**: Ein Agent, der DQN mit priorisiertem Experience Replay verwendet, um optimale Strategien zu lernen.
- **Priorisiertes Experience Replay**: Verbessert den Lernprozess, indem es sich auf die bedeutendsten Erfahrungen konzentriert.
- **Visualisierung**: Echtzeit-Rendering des Spielzustands und der Aktionen des Agenten.
- **Hyperparameter-Tuning**: Tools zum Experimentieren mit verschiedenen Hyperparametern, um die beste Konfiguration zu finden.

## Installation

Um dieses Projekt auszuführen, müssen Sie Python auf Ihrem Computer installiert haben. Sie können die erforderlichen Abhängigkeiten mit pip installieren:

```bash
pip install numpy pygame tensorflow matplotlib
```

## Nutzung

1. **Umgebung testen**: Führen Sie die Umgebung aus, um sicherzustellen, dass alles korrekt eingerichtet ist.

```bash
python tower_defense_rl.py
```

2. **Agent trainieren**: Starten Sie das Training des Agenten mit Standard- oder benutzerdefinierten Hyperparametern.

```bash
python tower_defense_rl.py --train --alpha 0.6 --beta_start 0.4
```

3. **Training visualisieren**: Die Umgebung rendert den Spielzustand und die Aktionen des Agenten während des Trainings. Sie können die Rendering-Frequenz anpassen, indem Sie den `RENDER_EVERY`-Parameter ändern.

## Konfiguration

Die Konfigurationsparameter sind zu Beginn des Skripts definiert. Sie können diese Parameter anpassen, um die Spielumgebung und den Trainingsprozess zu konfigurieren.

- **Spielkonfiguration**:
  - `WIDTH`, `HEIGHT`: Abmessungen des Spiel-Fensters.
  - `GRID_SIZE`: Größe des Spiel-Rasters.
  - `CELL_SIZE`: Größe jeder Zelle im Raster.
  - `NUM_EPISODES`: Anzahl der Trainings-Episoden.
  - `BATCH_SIZE`: Batch-Größe für Experience Replay.
  - `LEARNING_RATE`: Lernrate für das neuronale Netz.
  - `GAMMA`: Diskontfaktor für zukünftige Belohnungen.
  - `EPSILON_START`, `EPSILON_MIN`, `EPSILON_DECAY_RATE`: Parameter für epsilon-greedy Exploration.
  - `TARGET_UPDATE_FREQ`: Häufigkeit der Aktualisierung des Zielnetzwerks.
  - `MODEL_PATH`: Pfad zum Speichern des trainierten Modells.
  - `RENDER_EVERY`: Häufigkeit des Renderns des Spielzustands.
  - `PRIORITIZED_REPLAY_EPS`: Kleine Konstante für priorisiertes Experience Replay.
  - `MEMORY_SIZE_START`, `MEMORY_SIZE_MAX`: Anfangs- und maximale Größen des Replay-Puffers.

- **Turm- und Feindtypen**:
  - `TOWER_TYPES`: Liste der verfügbaren Turmtypen.
  - `ENEMY_TYPES`: Liste der verfügbaren Feindtypen.

- **Aktionen**:
  - `ACTIONS`: Liste der verfügbaren Aktionen für den Agenten.

- **Belohnungen und Strafen**:
  - `REWARD_ENEMY_KILL`: Belohnung für das Töten eines Feindes.
  - `REWARD_WAVE_COMPLETE`: Belohnung für das Abschließen einer Welle.
  - `REWARD_BASE_SURVIVAL`: Belohnung für das Überleben der Basis.
  - `PENALTY_ENEMY_HIT_BASE`: Strafe für einen Feind, der die Basis trifft.
  - `PENALTY_WASTED_RESOURCES`: Strafe für verschwendete Ressourcen.
  - `REWARD_STEP`: Belohnung für jeden Schritt.

## Training

Der Agent wird mit der `train_agent`-Funktion trainiert. Sie können die alpha- und beta_start-Parameter für priorisiertes Experience Replay angeben. Der Trainingsprozess umfasst die Echtzeit-Visualisierung des Spielzustands und der Aktionen des Agenten.

```python
def train_agent(alpha=0.6, beta_start=0.4):
    # Trainingslogik hier
```

## Ergebnisse

Nach dem Training wird die Leistung des Agenten anhand der durchschnittlichen Belohnung bewertet, die während der Episoden erzielt wurde. Die besten Hyperparameter werden durch eine Grid-Suche über verschiedene alpha- und beta_start-Werte ermittelt.

## Mitwirkung

Beiträge sind willkommen! Bitte öffnen Sie ein Issue oder senden Sie einen Pull Request, wenn Sie Vorschläge oder Verbesserungen haben.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details finden Sie in der [LICENSE](LICENSE)-Datei.

## Danksagungen

- Inspiriert von klassischen Tower Defense Spielen.
- Erstellt mit Python, Pygame, NumPy und TensorFlow.

Fühlen Sie sich frei, dieses Projekt nach Ihren Bedürfnissen anzupassen und zu erweitern. Viel Spaß beim Codieren!
