# Training Environment (PC-Side)

This directory contains the **machine learning training environment**.

## How to Run

```powershell
# Install uv (if not installed)
pip install uv

# Go to training environment directory path
cd ./training_pc

# Sync the environment (creates venv + installs deps)
uv sync
```

You're ready to train!

## To Learn

### Live Plot Sources

- [pySerial – Official Documentation](https://pyserial.readthedocs.io/en/latest/shortintro.html)
  Serielle Kommunikation mit dem ESP32, Einlesen von Daten über `readline()` und Dekodierung von Terminal-Ausgaben.

- [Python csv Module – Official Documentation](https://docs.python.org/3/library/csv.html)
  Strukturierte Speicherung von Messdaten in CSV-Dateien während einer laufenden Messung.

- [Matplotlib – Interactive Mode](https://matplotlib.org/stable/users/explain/interactive.html)
  Echtzeit-Visualisierung durch interaktiven Plot-Modus (`plt.ion()`, `plt.pause()`).

- [Matplotlib – Simple Animation Examples](https://matplotlib.org/stable/gallery/animation/simple_anim.html)
  Dynamisches Aktualisieren von Plot-Elementen während eines kontinuierlichen Datenstroms.

- [Python collections.deque – Official Documentation](https://docs.python.org/3/library/collections.html#collections.deque)
  Einsatz eines Sliding-Window-Buffers (`deque(maxlen=...)`) für stabile Live-Plots.
