#!/bin/bash

# Configuration
CHECK_INTERVAL=30
HEALTH_URL="http://localhost:52415/node_id"
LOG_FILE="exo.watchdog.log"
EXO_CMD=".venv/bin/python -m exo"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

start_exo() {
    log "Démarrage de exo..."
    # Lancement en arrière-plan, en redirigeant stdout/stderr pour éviter de polluer le terminal du watchdog
    $EXO_CMD > exo.log 2>&1 &
    EXO_PID=$!
    log "Exo démarré avec PID $EXO_PID"
    
    # Attendre un peu que le processus s'initialise
    sleep 5
}

cleanup() {
    log "Arrêt demandé. Arrêt du watchdog et de exo (PID $EXO_PID)..."
    if [ -n "$EXO_PID" ]; then
        kill $EXO_PID 2>/dev/null
    fi
    exit 0
}

# Piéger les signaux d'arrêt pour tuer proprement le processus fils
trap cleanup SIGINT SIGTERM

# Premier démarrage
start_exo

while true; do
    sleep $CHECK_INTERVAL
    
    # 1. Vérifier si le processus existe toujours
    if ! kill -0 $EXO_PID 2>/dev/null; then
        log "ALERTE: Processus exo ($EXO_PID) introuvable. Redémarrage..."
        start_exo
        continue
    fi

    # 2. Vérifier si l'API répond (Health Check)
    # On utilise curl avec un timeout court (5s) pour ne pas bloquer
    if ! curl -s --max-time 5 $HEALTH_URL > /dev/null; then
        log "ALERTE: L'API exo ne répond pas (Timeout ou Erreur). Redémarrage forcé..."
        kill $EXO_PID 2>/dev/null
        # On attend que le processus soit bien mort
        wait $EXO_PID 2>/dev/null
        start_exo
    else
        # Tout va bien
        # log "Status OK"
        :
    fi
done
