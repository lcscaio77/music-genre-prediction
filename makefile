# Variables
VENV = venv
PYTHON = python3

all: setup

# Crée un environnement virtuel
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	touch $(VENV)/bin/activate
	@echo "Environnement virtuel prêt, activez-le avec :" 
	@echo "Sur macOS/Linux : source $(VENV)/bin/activate ou $(VENV)\Scripts\activate"
	@echo "Sur Windows : nom_du_venv\Scripts\activate"

# Nettoie l'environnement
clean:
	rm -rf $(VENV)
	rm -f requirements.txt
	@echo "Environnement virtuel supprimé."
