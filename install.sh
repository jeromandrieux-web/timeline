#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Timeline · Install ==="

# 1) Vérif Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERREUR] python3 introuvable. Installe Python 3.x d'abord."
  exit 1
fi

# 2) Création du venv
if [ ! -d ".venv" ]; then
  echo "[INFO] Création de l'environnement virtuel .venv"
  python3 -m venv .venv
else
  echo "[INFO] Environnement .venv déjà présent"
fi

# 3) Activation
echo "[INFO] Activation de .venv"
# shellcheck disable=SC1091
source .venv/bin/activate

# 4) Installation dépendances
if [ ! -f "requirements.txt" ]; then
  echo "[ERREUR] requirements.txt introuvable dans le dossier courant."
  exit 1
fi

echo "[INFO] Installation / mise à jour des dépendances"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 5) .env par défaut si absent (version timeline)
if [ ! -f ".env" ]; then
  echo "[INFO] Création d'un .env par défaut"
  cat > .env << 'EOF'
# ============================
# APP
# ============================
APP_NAME=timeline-mvp
APP_ENV=dev
APP_DEBUG=true

HOST=127.0.0.1
PORT=8000

# ============================
# STORAGE
# ============================
DATA_DIR=./data
UPLOAD_DIR=./data/uploads
INDEX_DIR=./data/index
LOG_DIR=./logs

# ============================
# XML / TIMELINE
# ============================
XML_STRICT_MODE=true
XML_DEFAULT_TIMEZONE=Europe/Paris

# ============================
# PRIVACY
# ============================
ANONYMIZATION_ENABLED=true
ANONYMIZE_NAMES=true
ANONYMIZE_PHONE=true
ANONYMIZE_IP=false

# ============================
# (Optionnel) LLM/RAG plus tard
# ============================
LLM_PROVIDER=ollama
LLM_MODEL=mistral-large-3
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
EOF
else
  echo "[INFO] .env déjà présent, je ne le touche pas."
fi

# 6) Dossiers de travail
mkdir -p ./data/uploads ./data/index ./logs
echo "[INFO] Dossiers data/ et logs/ prêts."

# 7) Exécutables
chmod +x run.sh install.sh
echo "[INFO] run.sh et install.sh sont exécutables."

echo
echo "=== Installation terminée ==="
echo "Pour lancer le serveur :"
echo "  ./run.sh"
