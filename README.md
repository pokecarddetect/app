# 🃏 Pokémon Card Authentication App

Pokročilá Flask webová aplikace pro detekci autenticity Pokémon karet pomocí AI a počítačového vidění.

## 🌟 Funkce

- **AI Klasifikace**: Detekce originálních/falešných/proxy/custom art karet
- **OCR Analýza**: Rozpoznávání textu a detekce překlepů
- **Vícejazyčná podpora**: Čeština a angličtina
- **Uživatelské rozhraní**: Moderní Bootstrap design
- **Databáze analýz**: PostgreSQL s historií výsledků
- **Certifikáty**: Generování QR kódů a certifikátů autenticity

## 🚀 Rychlé spuštění

### Stažení z Replit
1. V Replit klikněte na tři tečky (...) → "Download as zip"
2. Rozbalte soubory na svém počítači

### Lokální spuštění
```bash
# Spusťte start script
./start.sh

# Nebo manuálně:
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -e .
python main.py
```

Aplikace běží na `http://localhost:5000`

## 🌐 Nasazení na cloud platformy

### 1. Heroku
```bash
heroku create your-app-name
heroku addons:create heroku-postgresql:hobby-dev
heroku config:set SESSION_SECRET=your-secret-key
git push heroku main
```

### 2. Railway
- Připojte GitHub repository
- Railway automaticky detekuje `railway.toml`
- Nastavte environment proměnné v dashboardu

### 3. DigitalOcean App Platform  
- Nahrajte na GitHub
- Vytvořte novou App a připojte repo
- App Platform použije `Procfile` automaticky

### 4. Docker
```bash
# Lokální build a spuštění
docker-compose up --build

# Nebo jen app bez databáze
docker build -t pokemon-app .
docker run -p 5000:5000 \
  -e DATABASE_URL=your-db-url \
  -e SESSION_SECRET=your-secret \
  pokemon-app
```

### 5. Vercel (Serverless)
```bash
vercel --prod
```

## ⚙️ Environment proměnné

Nutné nastavit pro produkci:
```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
SESSION_SECRET=your-super-secret-key
FLASK_ENV=production
```

## 📁 Struktura projektu

```
pokemon-card-app/
├── main.py              # Vstupní bod aplikace
├── app.py               # Flask konfigurace
├── routes.py            # Web routy
├── models.py            # Databázové modely
├── ai_model.py          # AI klasifikátor
├── templates/           # HTML šablony
├── static/              # CSS/JS soubory
├── deployment_guide.md  # Detailní nasazení návod
├── Dockerfile           # Docker konfigurace
├── Procfile             # Heroku konfigurace
├── requirements.txt     # Python závislosti
└── start.sh             # Rychlý start script
```

## 🛠️ Technologie

- **Backend**: Flask, SQLAlchemy, PostgreSQL
- **AI/ML**: OpenCV, Tesseract OCR, NumPy
- **Frontend**: Bootstrap, JavaScript
- **Deployment**: Gunicorn, Docker podporováno

## 📝 Poznámky pro produkci

- Aplikace používá lightweight fallback AI model (bez TensorFlow)
- Optimalizována pro nasazení - velikost pod 1GB
- PostgreSQL databáze nutná pro plnou funkcionalita
- Tesseract OCR potřeba nainstalovat na serveru

## 🔧 Troubleshooting

**Problém s databází**: Zkontrolujte `DATABASE_URL` a připojení k PostgreSQL

**OCR nefunguje**: Nainstalujte `tesseract-ocr` package

**Velká velikost**: Aplikace je optimalizovaná, ale pro další redukci odstraňte `certificates/` a `uploads/` adresáře

## 📞 Podpora

Pro detailní návod nasazení viz `deployment_guide.md`