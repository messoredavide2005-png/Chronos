# 🏛️ Chronos — Tutor di Storia AI

&gt; *"Più potente di NotebookLM. Tutto locale, tutto gratuito."*

Chronos è un tutor di storia intelligente che legge i tuoi PDF (libri di testo, appunti, dispense) e ti aiuta a studiare in **tre modalità** diverse. Costruisce automaticamente un **grafo di conoscenza** e una **timeline storica** dai tuoi documenti.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Ollama](https://img.shields.io/badge/Ollama-Local-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Feature

| Feature | Descrizione |
|---|---|
| 📚 **Tutor** | Risposte dirette con citazione delle fonti dal tuo PDF |
| 🎭 **Socrate** | Non ti dà la risposta: ti guida al ragionamento con domande mirate |
| 🎬 **Narratore** | Ti mette nel cuore degli eventi in **prima persona** (roleplay storico) |
| 🕵️ **Agente Storico** | Verifica che ogni fatt citato sia davvero nel documento (anti-allucinazione) |
| 📊 **Timeline Automatica** | Estrae istantaneamente date ed eventi, crea una linea del tempo visiva |
| 🕸️ **Grafo di Conoscenza** | Trova connessioni nascoste tra personaggi, luoghi e eventi |

---

## 🚀 Installazione

### 1. Prerequisiti
- Python 3.10+
- [Ollama](https://ollama.com) installato

### 2. Scarica i modelli AI
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
