# Variables
VENV = venv
PYTHON = python3

all: setup, activate

# Crée un environnement virtuel
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	touch $(VENV)/bin/activate
	@echo "Environnement virtuel prêt. Activez-le avec : source $(VENV)/bin/activate"

# Nettoie l'environnement
clean:
	rm -rf $(VENV)
	rm -f requirements.txt
	@echo "Environnement virtuel supprimé."

# Active l'environnement
activate:
	@echo "Pour activer : source $(VENV)/bin/activate"
